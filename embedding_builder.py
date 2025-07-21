from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import json
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(json_path, persist_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code_to_name = {entry['code']: entry['name'] for entry in data}
    reverse_map = defaultdict(list)
    for entry in data:
        for pre in entry.get("prerequisites", []):
            reverse_map[pre].append(entry["code"])

    docs = []
    for entry in data:
        code = entry.get('code', '').strip()
        name = entry.get('name', '').strip() or f"[Tên chưa rõ - mã {code}]"
        summary = entry.get('summary', '').strip()
        prerequisites = entry.get('prerequisites', [])
        clos = entry.get('clos', [])
        topics = entry.get('topics', [])
        instructors = entry.get('instructors', [])

        text_blocks = []

        # Thông tin cơ bản
        text_blocks.append(f"Học phần \"{name}\" (Mã: {code}) có mô tả: {summary}.")
        if clos:
            text_blocks.append(f"Sau khi học xong học phần \"{name}\", sinh viên sẽ đạt được: {'; '.join(clos)}.")

        # Thông tin topic
        if topics:
            topic_lines = []
            for t in topics:
                topic_lines.append(f"- Chủ đề \"{t['title']}\" (ID: {t.get('short_id', '')}), {t.get('theory_hours', 0)} tiết LT, {t.get('practice_hours', 0)} tiết TH.")
                if t.get("concepts"):
                    topic_lines.append(f"  + Các khái niệm: {', '.join(t['concepts'])}")
                if t.get("clos"):
                    topic_lines.append(f"  + Chuẩn đầu ra liên quan: {'; '.join(t['clos'])}")
            text_blocks.append(f"Các chủ đề được giảng dạy trong học phần \"{name}\" gồm:\n" + "\n".join(topic_lines))

        # Thông tin giảng viên
        if instructors:
            instructor_lines = []
            for ins in instructors:
                instructor_lines.append(f"- {ins['title']} {ins['name']} (Email: {ins['email']})")
            text_blocks.append(f"Học phần \"{name}\" được giảng dạy bởi:\n" + "\n".join(instructor_lines))

        # Tiên quyết
        if prerequisites:
            for pre in prerequisites:
                pre_name = code_to_name.get(pre, f"[Tên chưa rõ - mã {pre}]")
                text_blocks.append(f"Học phần \"{name}\" yêu cầu phải học trước học phần \"{pre_name}\" (Mã: {pre}).")
                text_blocks.append(f"Học phần \"{pre_name}\" là tiên quyết của học phần \"{name}\".")
        else:
            text_blocks.append(f"Học phần \"{name}\" không yêu cầu học phần tiên quyết nào.")

        # Hậu quyết
        post_reqs = reverse_map.get(code, [])
        for post in post_reqs:
            post_name = code_to_name.get(post, f"[Tên chưa rõ - mã {post}]")
            text_blocks.append(f"Học phần \"{name}\" là tiên quyết của học phần \"{post_name}\" (Mã: {post}).")

        # Tạo nội dung chính
        full_text = "\n".join(text_blocks)

        # Metadata (có thể mở rộng nếu muốn filter theo instructor)
        metadata = {
            "code": code,
            "name": name,
            "prerequisites": prerequisites,
            "clos": clos
        }

        docs.append(Document(page_content=full_text, metadata=metadata))

    # Cắt nhỏ nếu cần
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Tạo FAISS index
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(persist_path)

    print(f"✅ Đã lưu FAISS index tại: {persist_path} ({len(chunks)} chunks)")

# === Run khi gọi trực tiếp ===
if __name__ == "__main__":
    build_vectorstore("courses.json", "faiss_index")
