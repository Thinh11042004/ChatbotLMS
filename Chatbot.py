# Chatbot.py (Refactor toàn bộ với lựa chọn Prompt động, đánh giá, log, gợi ý)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from reranker import SentenceTransformersReranker
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from prompts import (
    QA_CHAIN_PROMPT,
    SUMMARY_PROMPT,
    STUDENT_SUPPORT_PROMPT,
    PREREQUISITE_PROMPT,
    CLO_EXPLAIN_PROMPT,
    ROADMAP_PROMPT,
    WARNING_PROMPT,
    COMPARISON_PROMPT,
    SKILL_PROMPT,
    WORKLOAD_PROMPT,
    TOPIC_LIST_PROMPT 
)


load_dotenv()
def choose_prompt(query: str):
    q = query.lower()
    if "tóm tắt" in q or "nội dung chính" in q:
        return SUMMARY_PROMPT
    elif "tiên quyết" in q or "môn trước" in q:
        return PREREQUISITE_PROMPT
    elif "chuẩn đầu ra" in q or "clo" in q:
        return CLO_EXPLAIN_PROMPT
    elif "tư vấn" in q or "lộ trình" in q:
        return ROADMAP_PROMPT
    elif "khó" in q or "cảnh báo" in q:
        return WARNING_PROMPT
    elif "so sánh" in q or "khác biệt" in q:
        return COMPARISON_PROMPT
    elif "kỹ năng" in q:
        return SKILL_PROMPT
    elif "nặng" in q or "tốn thời gian" in q:
        return WORKLOAD_PROMPT
    elif "năm nhất" in q or "gợi ý" in q:
        return STUDENT_SUPPORT_PROMPT
    elif "topic" in q or "chủ đề" in q or "nội dung" in q or "học gì" in q or "có trong học phần" in q:
        return TOPIC_LIST_PROMPT  
    else:
        return QA_CHAIN_PROMPT

# === Model & Embedding ===
llm = ChatOpenAI(temperature=0, model="gpt-4")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# === Load FAISS ===
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# === Reranker ===
reranker = SentenceTransformersReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever = db.as_retriever(search_kwargs={"k": 6})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever,
)

# === CLI Giao diện ===
print("\U0001f50d Đặt câu hỏi về học phần (gõ 'exit' để thoát):\n")

while True:
    query = input("🧑‍🏫 Bạn hỏi gì? > ").strip()
    if query.lower() == "exit":
        break

    prompt = choose_prompt(query)

    print("\n[1] 🔍 RETRIEVAL (Từ FAISS):")
    retrieved_docs = retrieved_docs = retriever.invoke(query)

    for i, doc in enumerate(retrieved_docs):
        print(f"--- Doc {i+1} ---\n{doc.page_content[:300]} ...\n")

    print("[2] 📊 RERANKED (Sau khi rerank):")
    reranked_docs = compression_retriever.invoke(query)

    for i, doc in enumerate(reranked_docs):
        print(f"⭐ Top {i+1}\n{doc.page_content[:300]} ...\n")

    # Chain khới tạo sau khi biết prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("[3] 🤖 LLM ANSWER:")
    result = qa_chain.invoke(query)
    print(result["result"])

    print("\n[4] 📌 Nguồn tài liệu ")
    for doc in result["source_documents"]:
        print("- " + doc.page_content[:150].replace("\n", " ") + "...")

    # === Logging ===
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== LOG {query} ===\n")
        f.write(f"Prompt: {prompt.template[:50]}...\n")
        f.write(f"RETRIEVED: {len(retrieved_docs)} docs\n")
        f.write(f"RERANKED TOP: {reranked_docs[0].page_content[:120]}...\n")
        f.write(f"ANSWER: {result['result']}\n")
        f.write("="*50 + "\n")

    print("\n" + "="*80 + "\n")
