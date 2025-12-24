import os
from dotenv import load_dotenv # 1. Library import karein
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 2. .env file se variables load karein
load_dotenv()

# --- Configuration ---
DB_PATH = "faiss_index"
# 3. Environment variable se key lein
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class MedicalBot:
    def __init__(self):
        # Embeddings loading
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # FAISS Database load karein
        if not os.path.exists(DB_PATH):
            raise Exception("FAISS index not found! Please run database_creator.py first.")
            
        self.db = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        
        # LLM setup
        if not OPENROUTER_API_KEY:
            raise Exception("API Key not found! Check your .env file.")

        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0 
        )

    def ask(self, query):
        docs = self.db.similarity_search(query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        full_prompt = f"""
        ROLE: You are a professional medical document assistant.
        
        INSTRUCTIONS:
        1. Use ONLY the provided context to answer.
        2. Format your response using Markdown (Headings, Bold text, Bullet points).
        3. If the answer is not in the context, strictly say you don't know.

        MEDICAL CONTEXT:
        {context_text}

        USER QUESTION: 
        {query}

        FINAL ANSWER:
        """
        
        response = self.llm.invoke(full_prompt)
        
        # FIX: Yahan double {{ ki jagah single { hona chahiye
        return {
            "answer": response.content,
            "context": docs
        }

def get_qa_chain():
    return MedicalBot()