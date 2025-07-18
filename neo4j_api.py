from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)  # Cho phép truy cập từ bên ngoài nếu bạn gọi từ chatbot

# Cấu hình kết nối Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Route chính để nhận Cypher và trả kết quả
@app.route("/query", methods=["POST"])
def run_query():
    data = request.json
    cypher = data.get("query", "")

    with driver.session() as session:
        result = session.run(cypher)
        records = [record.data() for record in result]

    return jsonify({"data": records})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
