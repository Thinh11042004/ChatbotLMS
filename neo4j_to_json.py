import json
from neo4j_conn import Neo4jConnection

conn = Neo4jConnection()
driver = conn.driver

def fetch_courses(tx):
    query = """
    MATCH (c:Course)
    OPTIONAL MATCH (c)-[:REQUIRES]->(prereq:Course)
    OPTIONAL MATCH (c)-[:HAS_CLO]->(clo:CLO)
    RETURN c.code AS code, c.name AS name, c.summary AS summary,
           collect(DISTINCT prereq.code) AS prerequisites,
           collect(DISTINCT clo.text) AS clos
    """
    return list(tx.run(query))

def get_all_courses():
    with driver.session() as session:
        result = session.read_transaction(fetch_courses)
        courses = []
        for record in result:
            courses.append({
                "code": record["code"],
                "name": record["name"],
                "summary": record["summary"] or "",
                "prerequisites": record["prerequisites"],
                "clos": record["clos"]
            })
        return courses

def export_to_json(path="courses.json"):
    courses = get_all_courses()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported {len(courses)} courses to {path}")

if __name__ == "__main__":
    export_to_json()
