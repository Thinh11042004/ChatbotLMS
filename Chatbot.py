# Chatbot.py (Refactor toàn bộ với lựa chọn Prompt động, đánh giá, log, gợi ý)
import os
import re
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from reranker import SentenceTransformersReranker
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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

    # === Tóm tắt / nội dung chính ===
    if re.search(r"(tóm tắt|nội dung chính|môn này nói về gì|giới thiệu)", q):
        return SUMMARY_PROMPT

    # === Môn tiên quyết ===
    elif re.search(r"(tiên quyết|môn.*trước|phải học.*trước|cần học.*trước)", q):
        return PREREQUISITE_PROMPT

    # === CLO / chuẩn đầu ra ===
    elif re.search(r"(chuẩn đầu ra|clo|đạt được gì|sau môn này)", q):
        return CLO_EXPLAIN_PROMPT

    # === Tư vấn lộ trình / môn học tiếp theo / học gì trước sau ===
    elif re.search(r"(tư vấn|lộ trình|học tiếp|nên học gì|xếp môn|thứ tự học)", q):
        return ROADMAP_PROMPT

    # === Cảnh báo độ khó / khối lượng cao ===
    elif re.search(r"(khó|cảnh báo|áp lực|môn nặng|có khó không)", q):
        return WARNING_PROMPT

    # === So sánh giữa các môn ===
    elif re.search(r"(so sánh|khác nhau|khác biệt|môn nào tốt hơn|môn nào dễ hơn)", q):
        return COMPARISON_PROMPT

    # === Kỹ năng sau học phần ===
    elif re.search(r"(kỹ năng|đạt được gì|ứng dụng|ra trường làm gì|môn này dùng để làm gì)", q):
        return SKILL_PROMPT

    # === Khối lượng học tập ===
    elif re.search(r"(khối lượng|tốn thời gian|nặng|có nhiều bài tập|phải học nhiều)", q):
        return WORKLOAD_PROMPT

    # === Tư vấn năm nhất / môn nên học sớm ===
    elif re.search(r"(năm nhất|gợi ý học|nên học trước|mới vào nên học)", q):
        return STUDENT_SUPPORT_PROMPT

    # === Danh sách topic / chủ đề / nội dung học / học gì ===
    elif re.search(r"(topic|chủ đề|nội dung|học gì|bao gồm|bài học|môn học này có gì)", q):
        return TOPIC_LIST_PROMPT

    # === Mặc định ===
    else:
        return QA_CHAIN_PROMPT
    
# === Model & Embedding ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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

    # # Chain khới tạo sau khi biết prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("[3] 🤖 LLM ANSWER:")
    result = qa_chain.invoke(query)
    print(result["result"])

    # print("\n[4] 📌 Nguồn tài liệu ")
    # for doc in result["source_documents"]:
    #     print("- " + doc.page_content[:150].replace("\n", " ") + "...")

    # # === Logging ===
    # with open("log.txt", "a", encoding="utf-8") as f:
    #     f.write(f"\n=== LOG {query} ===\n")
    #     f.write(f"Prompt: {prompt.template[:50]}...\n")
    #     f.write(f"RETRIEVED: {len(retrieved_docs)} docs\n")
    #     f.write(f"RERANKED TOP: {reranked_docs[0].page_content[:120]}...\n")
    #     f.write(f"ANSWER: {result['result']}\n")
    #     f.write("="*50 + "\n")

    # print("\n" + "="*80 + "\n")
