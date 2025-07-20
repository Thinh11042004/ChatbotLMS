# embedding_builder.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def build_vectorstore(json_path, persist_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for entry in data:
        text = (
            f"Mã học phần: {entry['code']}\n"
            f"Tên: {entry['name']}\n"
            f"Tóm tắt: {entry['summary']}\n"
            f"Tiên quyết: {', '.join(entry['prerequisites'])}\n"
            f"Chuẩn đầu ra: {'; '.join(entry['clos'])}"
        )
        docs.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(docs)

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(persist_path)
    print(f"✅ Saved embedding DB to: {persist_path}")

# Run
if __name__ == "__main__":
    build_vectorstore("courses.json", "faiss_index")
