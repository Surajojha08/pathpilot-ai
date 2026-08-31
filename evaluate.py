"""Offline benchmark for PathPilot's recommender.
Labels are expert-curated expected top-3 learning resources for representative learner states.
This is an internal validation benchmark, not a real-world accuracy claim.
"""
from app.main import Profile, recommend_resources
import math

CASES = [
    ("backend_java_beginner", Profile(goal="Backend Developer", experience="Beginner", skills=["Java"], interests=["backend"], hours_per_week=7), ["sql-101","rest-api","springboot"]),
    ("backend_java_intermediate", Profile(goal="Backend Developer", experience="Intermediate", skills=["Java"], interests=["backend"], hours_per_week=10), ["sql-101","rest-api","springboot"]),
    ("backend_java_sql", Profile(goal="Backend Developer", experience="Intermediate", skills=["Java","SQL"], interests=["backend"], hours_per_week=10), ["rest-api","springboot","jpa"]),
    ("backend_java_sql_docker", Profile(goal="Backend Developer", experience="Advanced", skills=["Java","SQL","REST APIs","Spring Boot","JPA/Hibernate"], interests=["backend","deployment"], hours_per_week=12), ["docker","system-design","backend-project"]),
    ("backend_sql_only", Profile(goal="Backend Developer", experience="Intermediate", skills=["SQL"], interests=["backend"], hours_per_week=8), ["rest-api","springboot","jpa"]),
    ("backend_no_skills", Profile(goal="Backend Developer", experience="Beginner", skills=[], interests=["backend"], hours_per_week=6), ["java-101","sql-101","rest-api"]),
    ("ai_python", Profile(goal="AI/ML Engineer", experience="Intermediate", skills=["Python"], interests=["AI","ML"], hours_per_week=10), ["ml","deep-learning","llm"]),
    ("ai_python_advanced", Profile(goal="AI/ML Engineer", experience="Advanced", skills=["Python","Machine Learning"], interests=["AI","LLM"], hours_per_week=12), ["deep-learning","llm","rag-project"]),
    ("ai_beginner", Profile(goal="AI/ML Engineer", experience="Beginner", skills=[], interests=["AI"], hours_per_week=8), ["python","ml","deep-learning"]),
    ("ai_ml_done", Profile(goal="AI/ML Engineer", experience="Advanced", skills=["Python","Machine Learning","Deep Learning"], interests=["generative ai"], hours_per_week=10), ["llm","rag-project"]),
    ("frontend_html", Profile(goal="Frontend Developer", experience="Intermediate", skills=["HTML/CSS"], interests=["web"], hours_per_week=8), ["javascript","react","frontend-project"]),
    ("frontend_beginner", Profile(goal="Frontend Developer", experience="Beginner", skills=[], interests=["web","design"], hours_per_week=7), ["html-css","javascript","react"]),
    ("frontend_js", Profile(goal="Frontend Developer", experience="Intermediate", skills=["HTML/CSS","JavaScript"], interests=["react","web"], hours_per_week=10), ["react","frontend-project"]),
    ("frontend_react", Profile(goal="Frontend Developer", experience="Advanced", skills=["HTML/CSS","JavaScript","React"], interests=["frontend"], hours_per_week=10), ["frontend-project"]),
    ("devops_linux", Profile(goal="DevOps Engineer", experience="Intermediate", skills=["Linux"], interests=["cloud"], hours_per_week=8), ["docker","cicd","cloud"]),
    ("devops_beginner", Profile(goal="DevOps Engineer", experience="Beginner", skills=[], interests=["devops"], hours_per_week=7), ["linux","docker","cicd"]),
    ("devops_docker", Profile(goal="DevOps Engineer", experience="Intermediate", skills=["Linux","Docker"], interests=["deployment","cloud"], hours_per_week=10), ["cicd","cloud"]),
    ("devops_cloud", Profile(goal="DevOps Engineer", experience="Advanced", skills=["Linux","Docker","CI/CD"], interests=["cloud"], hours_per_week=12), ["cloud"]),
]

def precision_at_k(pred, truth, k):
    return sum(x in truth for x in pred[:k]) / k

def recall_at_k(pred, truth, k):
    return sum(x in truth for x in pred[:k]) / min(k, len(truth))

def ndcg_at_k(pred, truth, k):
    rel={x:len(truth)-i for i,x in enumerate(truth)}
    dcg=sum(rel.get(x,0)/math.log2(i+2) for i,x in enumerate(pred[:k]))
    ideal=sum(v/math.log2(i+2) for i,v in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return dcg/ideal if ideal else 0

rows=[]
for name,p,truth in CASES:
    out=recommend_resources(p)
    pred=[r['id'] for r in out['resources']]
    rows.append({"case":name,"P@3":precision_at_k(pred,truth,3),"R@3":recall_at_k(pred,truth,3),"NDCG@3":ndcg_at_k(pred,truth,3),"top3":pred[:3]})

for r in rows: print(r)
print("AVERAGES")
for m in ["P@3","R@3","NDCG@3"]: print(m, round(sum(r[m] for r in rows)/len(rows),4))
