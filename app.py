import streamlit as st
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Optional

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb


class PrivateRAGChatbot:
    """Privacy-beschermde RAG Chatbot voor bedrijfskennisbank"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_components()
    
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
                temperature=0.1  # Consistente antwoorden
            )
            
            # Nederlandse embeddings
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/distiluse-base-multilingual-cased"
            )
            
            # Global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            
            # Privacy & security settings
            self.access_password = st.secrets.get("access_password", "")
            self.authorized_emails = st.secrets.get("authorized_emails", [])
            
            st.success("✅ Systeem succesvol geïnitialiseerd!")
            
        except Exception as e:
            st.error(f"❌ Fout bij initialisatie: {str(e)}")
            st.stop()
    
    def check_access_control(self) -> bool:
        """Controleer toegang tot de applicatie"""
        if not self.access_password:
            return True  # Geen toegangscontrole
        
        if 'authenticated' not in st.session_state:
            st.title("🔒 Toegang Vereist")
            
            with st.form("login_form"):
                password = st.text_input("Wachtwoord:", type="password")
                submit = st.form_submit_button("Inloggen")
                
                if submit:
                    if password == self.access_password:
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
    
    def process_uploaded_documents(self, uploaded_file) -> tuple[List[Document], str]:
        """Verwerk uploaded JSON bestand veilig"""
        try:
            # Hash voor privacy
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            
            # Parse JSON
            raw_data = json.loads(uploaded_file.getvalue())
            
            if not isinstance(raw_data, list):
                return [], "❌ JSON moet een lijst van documenten zijn"
            
            documents = []
            processed_count = 0
            
            for i, item in enumerate(raw_data):
                try:
                    # Verwachte velden
                    title = item.get('title', f'Document {i+1}')
                    content = item.get('content', '')
                    category = item.get('category', 'Algemeen')
                    audience = item.get('audience', 'Algemeen')
                    tags = item.get('tags', [])
                    
                    # Sanitize content
                    clean_content = self.sanitize_content(content)
                    
                    # Enhanced document text
                    doc_text = f"""
TITEL: {title}

INHOUD:
{clean_content}

CATEGORIE: {category}
DOELGROEP: {audience}
TAGS: {', '.join(tags) if tags else 'Geen tags'}
"""
                    
                    # Metadata
                    metadata = {
                        "doc_id": f"{file_hash}_{i}",
                        "title": title,
                        "category": category,
                        "audience": audience,
                        "tags": tags,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    documents.append(Document(text=doc_text.strip(), metadata=metadata))
                    processed_count += 1
                    
                except Exception as e:
                    st.warning(f"⚠️ Fout bij verwerken document {i+1}: {str(e)}")
                    continue
            
            if processed_count == 0:
                return [], "❌ Geen documenten konden worden verwerkt"
            
            success_msg = f"✅ {processed_count} documenten succesvol verwerkt (ID: {file_hash})"
            return documents, success_msg
            
        except json.JSONDecodeError:
            return [], "❌ Ongeldig JSON bestand"
        except Exception as e:
            return [], f"❌ Fout bij verwerken: {str(e)}"
    
    def create_vector_index(self, documents: List[Document]) -> Optional[VectorStoreIndex]:
        """Maak vector index van documenten"""
        try:
            with st.spinner("🔄 Vector index bouwen..."):
                # In-memory ChromaDB (privacy-vriendelijk)
                chroma_client = chromadb.EphemeralClient()
                collection = chroma_client.get_or_create_collection(
                    name="private_knowledge_base"
                )
                
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Build index
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True
                )
                
                return index
                
        except Exception as e:
            st.error(f"❌ Fout bij maken vector index: {str(e)}")
            return None
    
    def create_query_engine(self, index: VectorStoreIndex):
        """Maak query engine met privacy controls"""
        
        privacy_system_prompt = """
        JE BENT EEN NEDERLANDSE KLANTENSERVICE ASSISTENT.
        
        BELANGRIJKE REGELS:
        - Antwoord ALTIJD in het Nederlands
        - Gebruik ALLEEN informatie uit de gegeven context
        - Als je geen relevant antwoord kunt vinden, zeg dat eerlijk
        - Verzin NOOIT informatie die niet in de context staat
        - Wees vriendelijk en professioneel
        - Geef concrete, bruikbare informatie
        
        PRIVACY REGELS:
        - Deel geen gevoelige bedrijfsinformatie
        - Refereer niet naar specifieke systemen of processen tenzij relevant
        - Houd antwoorden algemeen maar nuttig
        """
        
        return index.as_query_engine(
            similarity_top_k=3,
            streaming=False,
            system_prompt=privacy_system_prompt,
            response_mode="compact"
        )
    
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
            if 'query_engine' in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Zoeken in kennisbank..."):
                        try:
                            response = st.session_state.query_engine.query(prompt)
                            answer = str(response)
                            
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                        except Exception as e:
                            error_msg = f"❌ Er ging iets mis: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                with st.chat_message("assistant"):
                    warning_msg = "⚠️ Er is nog geen kennisbank geladen. Upload eerst je JSON bestand via de sidebar."
                    st.warning(warning_msg)
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    
    def display_sidebar(self):
        """Sidebar met upload en informatie"""
        with st.sidebar:
            st.header("📁 Kennisbank Management")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Kennisbank (JSON)",
                type="json",
                help="Upload je private kennisbank JSON bestand"
            )
            
            if uploaded_file:
                with st.spinner("🔐 Veilig verwerken..."):
                    documents, result_msg = self.process_uploaded_documents(uploaded_file)
                    
                    if documents:
                        st.success(result_msg)
                        
                        # Create vector index
                        index = self.create_vector_index(documents)
                        
                        if index:
                            # Create query engine
                            query_engine = self.create_query_engine(index)
                            
                            # Store in session
                            st.session_state['index'] = index
                            st.session_state['query_engine'] = query_engine
                            st.session_state['documents'] = documents
                            
                            st.success("🎯 Kennisbank succesvol geladen!")
                            
                            # Show stats
                            st.subheader("📊 Statistieken")
                            categories = {}
                            for doc in documents:
                                cat = doc.metadata.get('category', 'Onbekend')
                                categories[cat] = categories.get(cat, 0) + 1
                            
                            for cat, count in categories.items():
                                st.text(f"• {cat}: {count} documenten")
                        
                        else:
                            st.error("❌ Fout bij maken vector index")
                    else:
                        st.error(result_msg)
            
            # Privacy info
            st.subheader("🔒 Privacy Info")
            st.info("""
            **Privacy Maatregelen:**
            - ✅ Session-based opslag
            - ✅ Gevoelige data filtering  
            - ✅ Toegangscontrole
            - ✅ Geen permanente opslag
            - ✅ Lokale embedding processing
            """)
            
            # Clear data button
            if st.button("🧹 Clear Kennisbank"):
                for key in ['index', 'query_engine', 'documents']:
                    if key in st.session_state:
                        del st.session_state[key]
                if 'messages' in st.session_state:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Kennisbank is gewist. Upload een nieuw bestand om te beginnen."}
                    ]
                st.rerun()
    
    def display_header(self):
        """App header met informatie"""
        st.title("🔒 Private Kennisbank Chatbot")
        st.markdown("*Semi-private • Toegangscontrole • Bedrijfsdata*")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'query_engine' in st.session_state:
                st.success("🟢 Kennisbank Actief")
            else:
                st.warning("🟡 Geen Kennisbank")
        
        with col2:
            st.info("🔒 Privacy Mode")
        
        with col3:
            st.info("🌐 Online Hosting")
        
        with col4:
            if st.session_state.get('authenticated', True):
                st.success("🔓 Toegang OK")
            else:
                st.error("🔐 Geen Toegang")
    
    def run(self):
        """Hoofdfunctie om de app te starten"""
        
        # Access control
        if not self.check_access_control():
            return
        
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
        st.caption("🔒 Private RAG Chatbot | Gebouwd met Streamlit & LlamaIndex | Data blijft veilig")


# Main execution
def main():
    """Main functie"""
    try:
        chatbot = PrivateRAGChatbot()
        chatbot.run()
        
    except Exception as e:
        st.error(f"❌ Kritieke fout: {str(e)}")
        st.info("Neem contact op met de beheerder.")


if __name__ == "__main__":
    main()