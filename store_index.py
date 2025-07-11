
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV=os.getenv("PINECONE_API_ENV")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(
    api_key=PINECONE_API_KEY)

index ="medical-chatbot"

# Create Embedding for Each of the Text Vhunks & storing

docsearch = LangchainPinecone.from_texts([t.page_content for t in text_chunks], embedding=embeddings, index_name=index)