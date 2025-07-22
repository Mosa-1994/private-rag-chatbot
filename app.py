import streamlit as st
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LlamaIndex en SentenceTransformer imports
from llama_index.llms.groq import Groq
from sentence_transformers import SentenceTransformer


class StreamlitCompatibleRAG:
    """
    RAG Chatbot geoptimaliseerd voor Streamlit Cloud.
    - Gebruikt in-memory vector search met NumPy voor snelheid.
    - Geen schijfafhankelijkheden (zoals ChromaDB/SQLite).
    - State management via st.session_state voor een naadloze ervaring.
    - Focus op privacy met lokale embeddings en content filtering.
    """

    # --- Configuratie Constanten ---
    LLM_MODEL = "llama3-70b-8192"
    EMBEDDING_MODEL = 'sentence-transformers/distiluse-base-multilingual-cased'
    SIMILARITY_THRESHOLD = 0.3  # Drempelwaarde om bronnen te tonen

    def __init__(self):
        """Initialiseer de applicatie en de benodigde componenten."""
        self.setup_page_config()
        self.llm, self.embedder = self.initialize_components()

        # Gebruik st.session_state als de 'single source of truth'
        self.documents_data = st.session_state.get('documents_data', [])

    def setup_page_config(self):
        """Configureer de Streamlit pagina."""
        st.set_page_config(
            page_title="🔒 Private Kennisbank Chat",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    @st.cache_resource
    def initialize_components(_self):
        """
        Initialiseer en cache de zware componenten (LLM en embedder).
        Gebruik @_self omdat @st.cache_resource geen 'self' accepteert.
        """
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY")
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY niet gevonden in Streamlit secrets!")
                st.stop()

            llm = Groq(
                model=_self.LLM_MODEL,
                api_key=groq_api_key,
                temperature=0.1
            )
            
            embedder = SentenceTransformer(_self.EMBEDDING_MODEL)
            
            st.toast("✅ Systeemcomponenten succesvol geladen!", icon="🚀")
            return llm, embedder

        except Exception as e:
            st.error(f"❌ Kritieke fout bij initialisatie: {str(e)}")
            st.stop()

    def check_access_control(self) -> bool:
        """Controleer de toegang tot de applicatie via een wachtwoord."""
        access_password = st.secrets.get("access_password")
        if not access_password:
            return True  # Geen wachtwoord ingesteld, toegang verleend

        if 'authenticated' not in st.session_state:
            st.session_state['authenticated'] = False

        if st.session_state['authenticated']:
            return True

        st.title("🔒 Toegang Vereist")
        with st.form("login_form"):
            password = st.text_input("Wachtwoord:", type="password")
            submit = st.form_submit_button("Inloggen")

            if submit:
                if password == access_password:
                    st.session_state['authenticated'] = True
                    st.success("✅ Toegang verleend!")
                    st.rerun()
                else:
                    st.error("❌ Onjuist wachtwoord")
        
        st.info("💡 Neem contact op met de beheerder voor toegang.")
        return False

    def sanitize_content(self, content: str) -> str:
        """Verwijder basis gevoelige informatie uit content met regex."""
        if not content:
            return ""
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        content = re.sub(r'\b(?:\+31|0031|0)[0-9]{8,9}\b', '[TELEFOON]', content)
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', content)
        content = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[IP]', content)
        return content

    def process_uploaded_documents(self, uploaded_file):
        """Verwerk een geüpload JSON-bestand, maak embeddings en sla op in session_state."""
        try:
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            raw_data = json.loads(uploaded_file.getvalue())

            if not isinstance(raw_data, list):
                st.error("❌ JSON moet een lijst van document-objecten zijn.")
                return

            documents = []
            progress_bar = st.progress(0, "Verwerken van documenten...")

            for i, item in enumerate(raw_data):
                try:
                    title = item.get('title', f'Document {i+1}')
                    content = self.sanitize_content(item.get('content', ''))
                    full_text = f"Titel: {title}. Inhoud: {content}"
                    
                    documents.append({
                        "id": f"{file_hash}_{i}",
                        "title": title,
                        "content": content,
                        "category": item.get('category', 'Algemeen'),
                        "audience": item.get('audience', 'Algemeen'),
                        "tags": item.get('tags', []),
                        "embedding": self.embedder.encode(full_text),
                        "processed_at": datetime.now().isoformat()
                    })
                    progress_bar.progress((i + 1) / len(raw_data), f"Document {i+1}/{len(raw_data)} verwerkt...")
                except Exception as e:
                    st.warning(f"⚠️ Kon document {i+1} niet verwerken: {str(e)}")

            progress_bar.empty()

            if not documents:
                st.error("❌ Geen documenten konden worden verwerkt uit het bestand.")
                return

            # Sla alles op in session_state
            st.session_state['documents_data'] = documents
            st.session_state['document_embeddings_matrix'] = np.vstack([doc['embedding'] for doc in documents])
            st.session_state['last_processed_file'] = f"{uploaded_file.name}_{uploaded_file.size}"
            
            st.success(f"✅ {len(documents)} documenten succesvol verwerkt (ID: {file_hash})")
            st.rerun()

        except json.JSONDecodeError:
            st.error("❌ Ongeldig JSON-bestand. Controleer de structuur.")
        except Exception as e:
            st.error(f"❌ Onverwachte fout bij verwerken: {str(e)}")

    def find_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Vind vergelijkbare documenten met een snelle, gevectoriseerde NumPy-operatie."""
        if 'document_embeddings_matrix' not in st.session_state:
            return []

        try:
            query_embedding = self.embedder.encode(query).reshape(1, -1)
            
            # Bereken similariteit met de hele matrix in één keer
            all_similarities = cosine_similarity(query_embedding, st.session_state['document_embeddings_matrix'])[0]
            
            # Krijg de indices van de top_k hoogste scores
            # Gebruik argpartition voor betere performance dan argsort op grote datasets
            if len(all_similarities) > top_k:
                top_k_indices = np.argpartition(all_similarities, -top_k)[-top_k:]
                sorted_indices = top_k_indices[np.argsort(all_similarities[top_k_indices])][::-1]
            else:
                sorted_indices = np.argsort(all_similarities)[::-1]

            # Maak de resultatenlijst
            similar_docs = []
            for index in sorted_indices:
                doc = self.documents_data[index]
                similar_docs.append({**doc, "similarity": float(all_similarities[index])})
            
            return similar_docs
            
        except Exception as e:
            st.error(f"❌ Fout bij zoeken naar documenten: {str(e)}")
            return []

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Genereer een antwoord met de LLM op basis van de gevonden context."""
        if not context_docs:
            return "Ik heb hier nog geen informatie over in mijn kennisbank."

        context_parts = []
        for doc in context_docs:
            if doc['similarity'] > self.SIMILARITY_THRESHOLD:
                context_parts.append(f"--- BRON ---\nTITEL: {doc['title']}\nINHOUD: {doc['content'][:1500]}...\n")
        
        if not context_parts:
             return "Er is wat informatie, maar die sluit nog niet goed aan. Stel je je vraag even op een andere manier? Vind ik fijn."

        context = "\n".join(context_parts)
        prompt = f"""
ROL:
Je bent een deskundige, vriendelijke en professionele Nederlandstalige klantenservice-assistent.

BRONGEBRUIK:
- Baseer je antwoorden **uitsluitend** op de informatie in de onderstaande context.
- **Verzin nooit** informatie die niet in de context staat.
- Gebruik de context om een antwoord te geven dat direct aansluit bij de vraag van de gebruiker.
- Staat het antwoord niet in de context? Laat dit dan duidelijk weten.

REFERENTIE:
- Verwijs waar mogelijk naar de titel van het bronbestand. Bijvoorbeeld: "Volgens het document 'Installatiegids' kan je...".

ANTWOORDSTIJL:
- Geef een duidelijk, beknopt en direct antwoord op de vraag.
- Gebruik een professionele, maar toegankelijke toon.

CONTEXT:
{context}
VRAAG:
{query}
ANTWOORD:

ANTWOORD:
"""
        try:
            response = self.llm.complete(prompt)
            return str(response)
        except Exception as e:
            return f"❌ Fout bij het genereren van het antwoord: {str(e)}"

    def display_sidebar(self):
        """Toon de sidebar voor bestandsbeheer en informatie."""
        with st.sidebar:
            st.header("📁 Kennisbank Management")

            uploaded_file = st.file_uploader(
                "Upload Kennisbank (JSON)", type="json",
                help="Upload een JSON-bestand met een lijst van documenten. Elk document moet 'title' en 'content' bevatten."
            )

            if uploaded_file:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get('last_processed_file') != file_id:
                    self.process_uploaded_documents(uploaded_file)
                else:
                    st.info("✅ Dit bestand is al geladen.")

            if self.documents_data:
                st.subheader("📊 Kennisbank Status")
                st.metric("Aantal Documenten", f"{len(self.documents_data)}")
                
                if st.button("🧹 Kennisbank Wissen"):
                    keys_to_delete = ['documents_data', 'document_embeddings_matrix', 'last_processed_file', 'messages']
                    for key in keys_to_delete:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            st.subheader("🔒 Privacy & Techniek")
            st.info(
                "**Privacy-First:** Uw data wordt lokaal in uw browser-sessie verwerkt en niet opgeslagen op een server. "
                "Embeddings worden lokaal gegenereerd."
            )
            st.markdown(
                f"""
                - **LLM:** Groq `{self.LLM_MODEL}`
                - **Embeddings:** `{self.EMBEDDING_MODEL}`
                - **Vector Store:** In-Memory (NumPy)
                """
            )

    def display_chat_interface(self):
        """Toon de hoofd-chatinterface."""
        st.title("🔒 Private Kennisbank Chatbot")
        st.markdown("*Een veilige RAG-chatbot, compatibel met Streamlit Cloud, zonder externe database.*")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "👋 Hallo! Upload een kennisbank via de sidebar om te beginnen."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Stel uw vraag..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if not self.documents_data:
                    warning_msg = "⚠️ Geen kennisbank geladen. Upload eerst een bestand."
                    st.warning(warning_msg)
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                    return

                with st.spinner("🧠 Denken..."):
                    similar_docs = self.find_similar_documents(prompt)
                    response = self.generate_response(prompt, similar_docs)
                    st.markdown(response)
                    
                    # Toon bronnen in een expander
                    valid_sources = [doc for doc in similar_docs if doc['similarity'] > self.SIMILARITY_THRESHOLD]
                    if valid_sources:
                        with st.expander("📚 Gebruikte bronnen"):
                            for doc in valid_sources:
                                st.write(f"• **{doc['title']}** (Relevantie: {doc['similarity']:.1%})")
                
                st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        """Start de applicatie."""
        if self.check_access_control():
            self.display_sidebar()
            self.display_chat_interface()

# --- Hoofduitvoering ---
if __name__ == "__main__":
    try:
        app = StreamlitCompatibleRAG()
        app.run()
    except Exception as e:
        st.error(f"❌ Een onverwachte kritieke fout is opgetreden: {e}")
        st.exception(e)