# 🧠 Medical Chatbot using LangChain, Pinecone & HuggingFace

A medical question-answering chatbot powered by:

- LangChain
- Pinecone Vector Database
- HuggingFace models (`flan-t5-small`)
- Flask (Web Server)
- Transformers & Sentence Embeddings

---

## 🚀 Features

- Extracts and splits data from PDFs
- Embeds and stores them in Pinecone
- Uses HuggingFace model locally for inference
- Exposes chatbot over web UI via Flask

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SBAK729/Medical_Chatbot.git
cd medical-chatbot
```

# Create and Activate a Virtual Environment

conda create -n mchatbot python=3.10
conda activate mchatbot

# Install Dependencies

# Prepare Environment Variables

PINECONE_API_KEY=your-pinecone-key
PINECONE_API_ENV=your-env-name
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token

# Run the App Locally

python app.py

Visit http://localhost:5000 in your browser.
