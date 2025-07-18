import os
from dotenv import load_dotenv
import re
import requests  

from langchain_openai import ChatOpenAI
from prompts import (
    course_info_full_prompt,
    course_info_instructor_prompt,
    course_info_topic_prompt,
    lookup_course_prompt,
    requires_forward_prompt,
    requires_reverse_prompt,
    response_generation_prompt,
    friendly_chat_prompt
)

load_dotenv()
DEBUG = True

# === Cấu hình LLM ===
llm = ChatOpenAI(model="gpt-4o", temperature=0)




# === Gửi Cypher đến API Flask và nhận kết quả ===
def run_cypher_via_api(cypher_query: str):
    try:
        response = requests.post("http://localhost:5000/query", json={"query": cypher_query})
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print("⚠️ Lỗi khi gửi truy vấn đến API:", str(e))
        return []

# === Xác định hướng truy vấn ===
def detect_direction(question: str) -> str:
    q = question.lower()

    if any(kw in q for kw in [
        "giới thiệu", "dạy gì", "nội dung", "ai dạy", "giảng viên",
        "topic", "chuẩn đầu ra", "clo", "giáo viên", "được dạy bởi", "dạy bởi", "người dạy"
    ]) or re.search(r'\bcmp\d{3}\b', q, re.IGNORECASE):
        return "course_info"
    elif any(kw in q for kw in [
        "là môn gì", "môn học gì", "học phần gì",
        "giới thiệu", "nội dung", "môn gì",
        "có liên quan", "thuộc lĩnh vực", "liên quan đến", "trong lĩnh vực"
    ]):
        return "lookup"
    elif any(kw in q for kw in [
        "cần học", "điều kiện để học", "học gì trước",
        "trước khi học", "bắt buộc", "phải học gì", "tiên quyết", "yêu cầu"
    ]):
        return "requires"
    elif any(kw in q for kw in [
        "sau khi học", "tiếp theo học", "sau đó học",
        "có thể học", "tiếp theo là", "học gì tiếp"
    ]):
        return "followup"
    else:
        return "unknown"

# === Trả lời tùy vào nội dung được hỏi ===
def format_course_info(records, question=None):
    if not records:
        return "🤖 Bot: Mình không tìm thấy thông tin chi tiết cho học phần này trong cơ sở dữ liệu."

    r = records[0]
    name = r.get("course_name", "Không rõ")
    code = r.get("course_code", "")
    topics = r.get("topics", [])
    clos = r.get("CLOs", [])
    instructors = r.get("instructors", [])

    question = question.lower() if question else ""
    reply = ""

    # Các câu hỏi đặc biệt
    if any(kw in question for kw in ["giảng viên", "ai dạy", "người dạy", "dạy bởi"]):
        reply = f"👨‍🏫 Giảng viên học phần **{name} ({code})** là: {', '.join(instructors)}." if instructors else "🤖 Bot: Không tìm thấy giảng viên."
        reply += f"\n👉 Bạn có muốn biết thêm về **chủ đề** hoặc **chuẩn đầu ra** của môn này không?"
        return reply

    if any(kw in question for kw in ["chủ đề", "topic"]):
        reply = f"🔖 Chủ đề của học phần **{name}**:\n" + "\n".join([f"- {t}" for t in topics]) if topics else "🤖 Bot: Không tìm thấy chủ đề."
        reply += f"\n👉 Bạn muốn biết thêm về **giảng viên** hoặc **chuẩn đầu ra** của môn này không?"
        return reply

    if any(kw in question for kw in ["chuẩn đầu ra", "clo"]):
        reply = f"🎯 Chuẩn đầu ra của **{name}**:\n" + "\n".join([f"- {clo}" for clo in clos]) if clos else "🤖 Bot: Không có chuẩn đầu ra nào."
        reply += f"\n👉 Bạn muốn xem thêm **giảng viên** hoặc **các chủ đề** được dạy trong môn này không?"
        return reply

    # Trường hợp chung
    reply += f"📘 **{name}** ({code}) là một học phần bao gồm:\n"
    if topics:
        reply += "🔖 **Chủ đề**:\n" + "\n".join([f"- {t}" for t in topics]) + "\n"
    if clos:
        reply += "🎯 **Chuẩn đầu ra**:\n" + "\n".join([f"- {clo}" for clo in clos]) + "\n"
    if instructors:
        reply += "👨‍🏫 **Giảng viên**: " + ", ".join(instructors) + "\n"

    reply += f"\n👉 Bạn có muốn biết **môn học tiên quyết** hoặc **môn tiếp theo** sau môn này không?"
    return reply.strip()



# === Vòng lặp chatbot ===
def chatbot_loop():
    print("🤖 Chatbot học phần sẵn sàng. Gõ câu hỏi (hoặc 'exit'):")

    context_memory = {
        "last_course_code": None,
        "last_course_name": None,
        "last_direction": None,
        "last_question": None
    }

    while True:
        question = input("❓ Bạn: ").strip()
        if question.lower() in ['exit', 'quit']:
            print("👋 Tạm biệt và chúc bạn học tốt!")
            break

        # Ghi nhớ câu hỏi
        context_memory["last_question"] = question

        direction = detect_direction(question)

        # Nếu không nhận diện được → fallback sang Chat thân thiện
        if direction == "unknown":
            response_chain = friendly_chat_prompt | llm
            reply = response_chain.invoke({"question": question})
            print("🤖 Bot:", reply.content.strip() if hasattr(reply, "content") else str(reply).strip())
            continue

        # Gán prompt phù hợp
        if direction == "requires":
            prompt = requires_forward_prompt
        elif direction == "followup":
            prompt = requires_reverse_prompt
        elif direction == "lookup":
            prompt = lookup_course_prompt
        elif direction == "course_info":
            q = question.lower()
            if any(kw in q for kw in ["ai dạy", "giảng viên", "giáo viên", "người dạy", "dạy bởi"]):
                prompt = course_info_instructor_prompt
            elif any(kw in q for kw in ["chủ đề", "topic"]):
                prompt = course_info_topic_prompt
            elif any(kw in q for kw in ["chuẩn đầu ra", "clo"]):
                prompt = course_info_full_prompt
            else:
                prompt = course_info_full_prompt
        else:
            prompt = None

        try:
            # Tạo câu lệnh Cypher
            cypher_query = (prompt | llm).invoke({"question": question}).content.strip()
            if "```" in cypher_query:
                cypher_query = re.sub(r"```[a-zA-Z]*", "", cypher_query).replace("```", "").strip()

            if DEBUG:
                print("📤 Cypher query:\n", cypher_query)

            # Gửi đến API backend
            records = run_cypher_via_api(cypher_query)

            # Ghi nhớ course nếu có
            for r in records:
                context_memory["last_course_code"] = r.get("course_code", context_memory["last_course_code"])
                context_memory["last_course_name"] = r.get("course_name", context_memory["last_course_name"])

            if direction == "course_info":
                print(format_course_info(records, question, context_memory))
                continue

            # Trả lời các hướng khác
            response_chain = response_generation_prompt | llm
            answer = response_chain.invoke({
                "question": question,
                "records": str(records)
            })

            print("🤖 Bot:", answer.content.strip() if hasattr(answer, "content") else str(answer).strip())

        except Exception as e:
            print("⚠️ Lỗi:", str(e))


if __name__ == "__main__":
    chatbot_loop()
