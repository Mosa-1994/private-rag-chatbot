import streamlit as st
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LlamaIndex imports - alleen wat je gebruikt
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from sentence_transformers import SentenceTransformer


class StreamlitCompatibleRAG:
    """RAG Chatbot compatible met Streamlit Cloud (geen ChromaDB)"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_components()
        self.documents_data = []  # Store voor document data
        self.embeddings_cache = {}  # Cache voor embeddings
    
    def setup_page_config(self):
        """Streamlit pagina configuratie"""
        st.set_page_config(
            page_title="🔒 Private Kennisbank Chat",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        """Initialize LLM en embedding components"""
        try:
            # Groq LLM setup
            groq_api_key = st.secrets.get("GROQ_API_KEY")
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY niet gevonden in Streamlit secrets!")
                st.stop()
            
            self.llm = Groq(
                model="llama3-70b-8192",
                api_key=groq_api_key,
                temperature=0.1
            )
            
            # Lokale sentence transformer voor embeddings
            self.embedder = SentenceTransformer(
                'sentence-transformers/distiluse-base-multilingual-cased'
            )
            
            st.success("✅ Systeem succesvol geïnitialiseerd (ChromaDB-vrij)!")
            
        except Exception as e:
            st.error(f"❌ Fout bij initialisatie: {str(e)}")
            st.stop()
    
    def check_access_control(self) -> bool:
        """Controleer toegang tot de applicatie"""
        access_password = st.secrets.get("access_password", "")
        
        if not access_password:
            return True  # Geen toegangscontrole
        
        if 'authenticated' not in st.session_state:
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
        
        return True
    
    def sanitize_content(self, content: str) -> str:
        """Verwijder gevoelige informatie uit content"""
        if not content:
            return ""
        
        # Email adressen
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # Telefoonnummers
        content = re.sub(r'\b(?:\+31|0031|0)[0-9]{8,9}\b', '[TELEFOON]', content)
        
        # URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', content)
        
        # IP adressen
        content = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[IP]', content)
        
        return content
    
    def process_uploaded_documents(self, uploaded_file) -> tuple[List[Dict], str]:
        """Verwerk uploaded JSON bestand"""
        try:
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            raw_data = json.loads(uploaded_file.getvalue())
            
            if not isinstance(raw_data, list):
                return [], "❌ JSON moet een lijst van documenten zijn"
            
            documents = []
            processed_count = 0
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for i, item in enumerate(raw_data):
                try:
                    title = item.get('title', f'Document {i+1}')
                    content = item.get('content', '')
                    category = item.get('category', 'Algemeen')
                    audience = item.get('audience', 'Algemeen')
                    tags = item.get('tags', [])
                    
                    # Sanitize content
                    clean_content = self.sanitize_content(content)
                    
                    # Full document text voor embeddings
                    full_text = f"{title} {clean_content} {category}"
                    
                    # Create embedding
                    with st.spinner(f"Processing document {i+1}/{len(raw_data)}..."):
                        embedding = self.embedder.encode(full_text)
                    
                    document_data = {
                        "id": f"{file_hash}_{i}",
                        "title": title,
                        "content": clean_content,
                        "category": category,
                        "audience": audience,
                        "tags": tags,
                        "full_text": full_text,
                        "embedding": embedding,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    documents.append(document_data)
                    processed_count += 1
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(raw_data))
                    
                except Exception as e:
                    st.warning(f"⚠️ Fout bij verwerken document {i+1}: {str(e)}")
                    continue
            
            progress_bar.empty()
            
            if processed_count == 0:
                return [], "❌ Geen documenten konden worden verwerkt"
            
            success_msg = f"✅ {processed_count} documenten succesvol verwerkt (ID: {file_hash})"
            return documents, success_msg
            
        except json.JSONDecodeError:
            return [], "❌ Ongeldig JSON bestand"
        except Exception as e:
            return [], f"❌ Fout bij verwerken: {str(e)}"
    
    def find_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Vind vergelijkbare documenten zonder ChromaDB"""
        if not self.documents_data:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedder.encode(query)
            
            # Calculate similarities
            similarities = []
            for doc in self.documents_data:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    doc["embedding"].reshape(1, -1)
                )[0][0]
                
                similarities.append({
                    **doc,
                    "similarity": float(similarity)
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            st.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Genereer response met Groq LLM"""
        
        if not context_docs:
            return "❌ Geen relevante informatie gevonden in de kennisbank."
        
        # Build context
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"""
TITEL: {doc['title']}
CATEGORIE: {doc['category']}
INHOUD: {doc['content'][:1000]}...
RELEVANTIE: {doc['similarity']:.1%}
---""")
        
        context = "\n".join(context_parts)
        
        # Privacy-aware prompt
        prompt = f"""
JE BENT EEN NEDERLANDSE KLANTENSERVICE ASSISTENT.

BELANGRIJKE REGELS:
- Antwoord ALTIJD in het Nederlands
- Gebruik ALLEEN informatie uit de onderstaande context
- Als je geen relevant antwoord kunt vinden, zeg dat eerlijk
- Verzin NOOIT informatie die niet in de context staat
- Wees vriendelijk en professioneel
- Geef concrete, bruikbare informatie
- Refereer naar de titel van relevante documenten

CONTEXT:
{context}

VRAAG: {query}

ANTWOORD:
"""
        
        try:
            # Call Groq API
            response = self.llm.complete(prompt)
            return str(response)
            
        except Exception as e:
            return f"❌ Fout bij genereren antwoord: {str(e)}"
    
    def display_chat_interface(self):
        """Hoofdchat interface"""
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "👋 Hallo! Ik ben je private kennisbank assistent. Upload eerst je kennisbank via de sidebar, dan kan ik je vragen beantwoorden."
                }
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Stel je vraag over de kennisbank..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            if self.documents_data:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Zoeken in kennisbank..."):
                        # Find similar documents
                        similar_docs = self.find_similar_documents(prompt)
                        
                        # Generate response
                        response = self.generate_response(prompt, similar_docs)
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show sources if available
                        if similar_docs and similar_docs[0]["similarity"] > 0.3:
                            with st.expander("📚 Gebruikte bronnen"):
                                for doc in similar_docs[:2]:
                                    st.write(f"• **{doc['title']}** ({doc['similarity']:.1%} relevant)")
            else:
                with st.chat_message("assistant"):
                    warning_msg = "⚠️ Er is nog geen kennisbank geladen. Upload eerst je JSON bestand via de sidebar."
                    st.warning(warning_msg)
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    
    def display_sidebar(self):
        """Sidebar met upload en informatie"""
        with st.sidebar:
            st.header("📁 Kennisbank Management")
            
            # Load from session if available (EERST controleren!)
            if 'documents_data' in st.session_state and not self.documents_data:
                self.documents_data = st.session_state['documents_data']
                st.info(f"✅ Kennisbank herlaadt: {len(self.documents_data)} documenten")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Kennisbank (JSON)",
                type="json",
                help="Upload je private kennisbank JSON bestand",
                key="json_uploader"
            )
            
            if uploaded_file:
                # Voorkom herverwerking van hetzelfde bestand
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                
                if st.session_state.get('last_processed_file') != file_id:
                    # Process documents
                    with st.spinner("📄 Verwerken van documenten..."):
                        documents, result_msg = self.process_uploaded_documents(uploaded_file)
                    
                    if documents:
                        st.success(result_msg)
                        
                        # Store in BOTH class and session
                        self.documents_data = documents
                        st.session_state['documents_data'] = documents
                        st.session_state['last_processed_file'] = file_id
                        
                        # Force rerun to update the interface
                        st.rerun()
                    else:
                        st.error(result_msg)
                else:
                    st.info("✅ Bestand al verwerkt")
            
            # Status check (DEBUG info)
            st.subheader("🔍 Debug Status")
            st.write(f"Documents in class: {len(self.documents_data)}")
            st.write(f"Documents in session: {len(st.session_state.get('documents_data', []))}")
            
            # Show current stats if data exists
            if self.documents_data:
                st.subheader("📊 Statistieken")
                categories = {}
                for doc in self.documents_data:
                    cat = doc.get('category', 'Onbekend')
                    categories[cat] = categories.get(cat, 0) + 1
                
                for cat, count in categories.items():
                    st.text(f"• {cat}: {count} documenten")
                
                # Show quality metrics
                avg_content_length = np.mean([len(doc['content']) for doc in self.documents_data])
                st.metric("Gem. artikel lengte", f"{avg_content_length:.0f} chars")
            
            # Load from session button (manual backup)
            if st.button("🔄 Herlaad uit Session"):
                if 'documents_data' in st.session_state:
                    self.documents_data = st.session_state['documents_data']
                    st.success(f"✅ {len(self.documents_data)} documenten herladen")
                    st.rerun()
                else:
                    st.error("❌ Geen data in session gevonden")
            
            # Privacy info
            st.subheader("🔒 Privacy Info")
            st.info("""
            **Streamlit Cloud Compatible:**
            - ✅ Geen ChromaDB (SQLite issue opgelost)
            - ✅ Session-based opslag
            - ✅ Gevoelige data filtering  
            - ✅ Lokale embeddings
            - ✅ Privacy-first design
            """)
            
            # System info
            st.subheader("⚙️ Technische Info")
            st.text("🤖 LLM: Groq Llama3-70B")
            st.text("🔍 Embeddings: SentenceTransformers")
            st.text("🗄️ Vector Store: In-Memory")
            st.text("☁️ Platform: Streamlit Cloud")
            
            # Clear data button
            if st.button("🧹 Clear Kennisbank"):
                self.documents_data = []
                if 'documents_data' in st.session_state:
                    del st.session_state['documents_data']
                if 'last_processed_file' in st.session_state:
                    del st.session_state['last_processed_file']
                if 'messages' in st.session_state:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Kennisbank is gewist. Upload een nieuw bestand om te beginnen."}
                    ]
                st.rerun()
    
    def display_header(self):
        """App header met informatie"""
        st.title("🔒 Private Kennisbank Chatbot")
        st.markdown("*Streamlit Cloud Compatible • ChromaDB-vrij • Privacy-first*")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.documents_data:
                st.success("🟢 Kennisbank Actief")
            else:
                st.warning("🟡 Geen Kennisbank")
        
        with col2:
            st.info("🔒 Privacy Mode")
        
        with col3:
            st.success("☁️ Cloud Compatible")
        
        with col4:
            if st.secrets.get("access_password", ""):
                if st.session_state.get('authenticated', False):
                    st.success("🔓 Toegang OK")
                else:
                    st.error("🔐 Geen Toegang")
            else:
                st.success("🔓 Open Access")
    
    def run(self):
        """Hoofdfunctie om de app te starten"""
        
        # Access control
        if not self.check_access_control():
            return
        
        # Load documents from session on startup (BELANGRIJK!)
        if 'documents_data' in st.session_state and not self.documents_data:
            self.documents_data = st.session_state['documents_data']
        
        # Display app
        self.display_header()
        
        # Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.display_chat_interface()
        
        with col2:
            self.display_sidebar()
        
        # Footer
        st.markdown("---")
        st.caption("🔒 Private RAG Chatbot | ChromaDB-vrij voor Streamlit Cloud | Privacy-first design")


# Main execution
def main():
    """Main functie"""
    try:
        chatbot = StreamlitCompatibleRAG()
        chatbot.run()
        
    except Exception as e:
        st.error(f"❌ Kritieke fout: {str(e)}")
        st.info("Neem contact op met de beheerder.")
        st.exception(e)  # Voor debugging


if __name__ == "__main__":
    main()