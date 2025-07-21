import json
from neo4j_conn import Neo4jConnection

conn = Neo4jConnection()
driver = conn.driver

def fetch_courses(tx):
    query = """
MATCH (c:Course)
OPTIONAL MATCH (c)-[:REQUIRES]->(prereq:Course)
OPTIONAL MATCH (c)-[:HAS_CLO]->(clo:CLO)
OPTIONAL MATCH (c)-[:TAUGHT_BY]->(ins:Instructor)
OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
OPTIONAL MATCH (t)-[:HAS_CONCEPT]->(con:Concept)
OPTIONAL MATCH (t)-[:COVERS]->(tclo:CLO)

WITH c, prereq, clo, ins, t, 
     collect(DISTINCT con.name) AS topic_concepts,
     collect(DISTINCT tclo.description) AS topic_clos

WITH c.code AS code, 
     c.name AS name, 
     c.summary AS summary,
     collect(DISTINCT prereq.code) AS prerequisites,
     collect(DISTINCT clo.description) AS clos,
     collect(DISTINCT {
         id: t.id,
         short_id: t.short_id,
         title: t.title,
         theory_hours: t.theory_hours,
         practice_hours: t.practice_hours,
         embedding: t.embedding,
         concepts: topic_concepts,
         clos: topic_clos
     }) AS topics,
     collect(DISTINCT {
         name: ins.name,
         email: ins.email,
         title: ins.title
     }) AS instructors

RETURN code, name, summary, prerequisites, clos, topics, instructors
    """
    return list(tx.run(query))

def get_all_courses():
    with driver.session() as session:
        result = session.execute_read(fetch_courses)
        courses = []
        for record in result:
            courses.append({
                "code": record["code"],
                "name": record["name"],
                "summary": record["summary"] or "",
                "prerequisites": record["prerequisites"],
                "clos": record["clos"],
                "topics": record["topics"],
                "instructors": record["instructors"]
            })
        return courses

def export_to_json(path="courses.json"):
    courses = get_all_courses()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported {len(courses)} courses to {path}")

if __name__ == "__main__":
    export_to_json()
