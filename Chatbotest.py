import os
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from reranker import SentenceTransformersReranker
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

# === Load biến môi trường ===
load_dotenv()

# === Hàm chọn Prompt ===
def choose_prompt(query: str):
    q = query.lower()
    if re.search(r"(tóm tắt|nội dung chính|môn này nói về gì|giới thiệu)", q):
        return SUMMARY_PROMPT
    elif re.search(r"(tiên quyết|môn.*trước|phải học.*trước|cần học.*trước)", q):
        return PREREQUISITE_PROMPT
    elif re.search(r"(chuẩn đầu ra|clo|đạt được gì|sau môn này)", q):
        return CLO_EXPLAIN_PROMPT
    elif re.search(r"(tư vấn|lộ trình|học tiếp|nên học gì|xếp môn|thứ tự học)", q):
        return ROADMAP_PROMPT
    elif re.search(r"(khó|cảnh báo|áp lực|môn nặng|có khó không)", q):
        return WARNING_PROMPT
    elif re.search(r"(so sánh|khác nhau|khác biệt|môn nào tốt hơn|môn nào dễ hơn)", q):
        return COMPARISON_PROMPT
    elif re.search(r"(kỹ năng|đạt được gì|ứng dụng|ra trường làm gì|môn này dùng để làm gì)", q):
        return SKILL_PROMPT
    elif re.search(r"(khối lượng|tốn thời gian|nặng|có nhiều bài tập|phải học nhiều)", q):
        return WORKLOAD_PROMPT
    elif re.search(r"(năm nhất|gợi ý học|nên học trước|mới vào nên học)", q):
        return STUDENT_SUPPORT_PROMPT
    elif re.search(r"(topic|chủ đề|nội dung|học gì|bao gồm|bài học|môn học này có gì)", q):
        return TOPIC_LIST_PROMPT
    else:
        return QA_CHAIN_PROMPT

# === LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# === Embedding & FAISS ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# === Reranker & Retriever ===
reranker = SentenceTransformersReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever = db.as_retriever(search_kwargs={"k": 6})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever,
)

# === Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# === CLI ===
print("🔍 Đặt câu hỏi về học phần (gõ 'exit' để thoát):\n")

while True:
    query = input("🧑‍🏫 Bạn hỏi gì? > ").strip()
    if query.lower() == "exit":
        break

    # --- Chọn Prompt phù hợp ---
    prompt = choose_prompt(query)

    # === Hiển thị tài liệu gốc từ FAISS ===
    print("\n[1] 🔍 RETRIEVAL (Từ FAISS):")
    retrieved_docs = retriever.invoke(query)
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Doc {i+1} ---\n{doc.page_content[:300]} ...\n")

    # === Sau khi rerank ===
    print("[2] 📊 RERANKED (Sau khi rerank):")
    reranked_docs = compression_retriever.invoke(query)
    for i, doc in enumerate(reranked_docs):
        print(f"⭐ Top {i+1}\n{doc.page_content[:300]} ...\n")

    # === Tạo history-aware retriever ===
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một trợ lý học tập, trả lời câu hỏi dựa trên bối cảnh hội thoại."),
        ("human", "{question}")
    ])


    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=compression_retriever,
        prompt=qa_prompt
    )

    # === Chain tổng hợp tài liệu bằng prompt động
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # === Tạo chain chính
    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    # === Gọi chain
    print("[3] 🤖 LLM ANSWER:")
    result = qa_chain.invoke({
        "question": query,
        "chat_history": memory.chat_memory.messages
    })
    print(result["answer"])

    # === Hiển thị nguồn
    print("\n[4] 📌 Nguồn tài liệu ")
    for doc in reranked_docs[:3]:
        print("- " + doc.page_content[:150].replace("\n", " ") + "...")

    # === Logging ra file
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== LOG {query} ===\n")
        f.write(f"Prompt: {prompt.template[:50]}...\n")
        f.write(f"RETRIEVED: {len(retrieved_docs)} docs\n")
        f.write(f"RERANKED TOP: {reranked_docs[0].page_content[:120]}...\n")
        f.write(f"ANSWER: {result['answer']}\n")
        f.write("="*50 + "\n")

    print("\n" + "="*80 + "\n")
