from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template
from langchain_huggingface import HuggingFaceEndpoint
import os



from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Initialize Flask app
app = Flask(__name__)

# Load .env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_API_ENV")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Unused for Pinecone v3

# 1. Load embeddings
embeddings = download_hugging_face_embeddings()


# 2. Initialize Pinecone v3 client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"  

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
# 4. Load the LangChain Pinecone VectorStore from index

docsearch = PineconeVectorStore(index=index, embedding=embeddings)

# 5. Load your prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 6. Set up HuggingFace model (Locally or switch to HF Inference API)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# 7. Set up LangChain QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

# 8. Flask route
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print(user_input)
    result = qa({"query":user_input})
    print("Response :", result["result"])
    return str(result["result"])

# 9. Run the app
if __name__ == '__main__':
    app.run(debug=True)
