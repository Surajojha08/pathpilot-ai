from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "resources.json"
DB_FILE = ROOT / "data" / "pathpilot.db"

with open(DATA_FILE, encoding="utf-8") as f:
    RESOURCES: list[dict[str, Any]] = json.load(f)

GOALS: dict[str, list[str]] = {
    "Backend Developer": ["Java", "SQL", "REST APIs", "Spring Boot", "JPA/Hibernate", "Docker", "System Design", "Backend Engineering"],
    "AI/ML Engineer": ["Python", "Machine Learning", "Deep Learning", "Generative AI"],
    "Frontend Developer": ["HTML/CSS", "JavaScript", "React", "Frontend Engineering"],
    "DevOps Engineer": ["Linux", "Docker", "CI/CD", "Cloud"],
}
GOAL_ALIASES = {
    "AI/ML Engineer": ["ai engineer", "machine learning engineer", "ml engineer", "machine learning", "artificial intelligence", "data scientist", "genai", "generative ai"],
    "Frontend Developer": ["frontend", "front end", "react developer", "frontend developer", "web developer", "ui developer"],
    "DevOps Engineer": ["devops", "cloud engineer", "site reliability", "sre", "platform engineer", "dev ops"],
    "Backend Developer": ["backend", "back end", "spring boot", "java developer", "api developer", "server side", "backend developer"],
}
SKILL_ALIASES = {
    "Java": ["java", "core java"], "SQL": ["sql", "postgres", "postgresql", "database"],
    "REST APIs": ["rest", "rest api", "rest apis", "api", "apis", "http"], "Spring Boot": ["spring boot", "springboot"],
    "JPA/Hibernate": ["jpa", "hibernate", "spring data jpa"], "Docker": ["docker", "container", "containers"],
    "System Design": ["system design", "architecture", "scalability"], "Backend Engineering": ["backend engineering"],
    "Python": ["python"], "Machine Learning": ["machine learning", "ml", "scikit learn", "scikit-learn"],
    "Deep Learning": ["deep learning", "pytorch", "tensorflow", "neural network"], "Generative AI": ["generative ai", "genai", "llm", "large language model", "rag"],
    "HTML/CSS": ["html", "css", "html css", "html/css"], "JavaScript": ["javascript", "js", "typescript"],
    "React": ["react", "reactjs"], "Frontend Engineering": ["frontend engineering"],
    "Linux": ["linux", "command line", "bash"], "CI/CD": ["ci/cd", "cicd", "github actions", "continuous integration"], "Cloud": ["cloud", "aws", "azure", "gcp"],
}
LEVELS = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}

app = FastAPI(title="PathPilot AI", version="3.0.0", description="Adaptive AI-powered personalized learning path recommender")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# ---------- Persistence ----------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS progress (
            learner_id TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'not_started',
            score REAL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (learner_id, resource_id)
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)


init_db()

# ---------- Models ----------
class Profile(BaseModel):
    learner_id: str = "demo-user"
    goal: str = "Backend Developer"
    experience: str = "Beginner"
    interests: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    hours_per_week: int = Field(default=7, ge=1, le=80)
    preference: str = "Balanced"


class ChatRequest(BaseModel):
    message: str
    profile: Profile


class ProgressRequest(BaseModel):
    learner_id: str
    resource_id: str
    status: str = Field(pattern="^(not_started|in_progress|completed)$")
    score: float | None = Field(default=None, ge=0, le=100)


class FeedbackRequest(BaseModel):
    learner_id: str
    resource_id: str
    action: str = Field(pattern="^(helpful|not_helpful|skip|save)$")


class AssessmentRequest(BaseModel):
    learner_id: str
    resource_id: str
    score: float = Field(ge=0, le=100)

# ---------- NLP / Recommendation ----------
def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9+#/ -]", " ", s.lower()).strip()


def find_goal(text: str) -> str | None:
    t = normalize(text)
    ranked = []
    for goal, aliases in GOAL_ALIASES.items():
        hits = sum(1 for a in aliases if normalize(a) in t)
        if hits:
            ranked.append((hits, max(len(a) for a in aliases if normalize(a) in t), goal))
    return max(ranked)[2] if ranked else None


def extract_skills(text: str) -> list[str]:
    t = normalize(text)
    found = []
    for skill, aliases in SKILL_ALIASES.items():
        if any(re.search(r"(?<![a-z0-9])" + re.escape(normalize(a)) + r"(?![a-z0-9])", t) for a in aliases):
            found.append(skill)
    return found


def canonical_skills(items: list[str]) -> set[str]:
    out: set[str] = set()
    for item in items:
        skills = extract_skills(item)
        out.update(skills or [item.strip()])
    return {x for x in out if x}


def resource_text(r: dict[str, Any]) -> str:
    return " ".join([r["title"], r["skill"], r["type"], r["level"], *r["tags"], *r["prerequisites"]])

VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
RESOURCE_MATRIX = VECTORIZER.fit_transform([resource_text(r) for r in RESOURCES])


def goal_requirements(goal: str) -> list[str]:
    return GOALS.get(goal, GOALS["Backend Developer"])


def feedback_map(learner_id: str) -> dict[str, int]:
    with db() as conn:
        rows = conn.execute("SELECT resource_id, action, COUNT(*) AS n FROM feedback WHERE learner_id=? GROUP BY resource_id, action", (learner_id,)).fetchall()
    signals = defaultdict(int)
    for r in rows:
        if r["action"] == "helpful": signals[r["resource_id"]] += 2 * r["n"]
        elif r["action"] == "save": signals[r["resource_id"]] += r["n"]
        elif r["action"] in {"skip", "not_helpful"}: signals[r["resource_id"]] -= 2 * r["n"]
    return dict(signals)


def progress_map(learner_id: str) -> dict[str, dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM progress WHERE learner_id=?", (learner_id,)).fetchall()
    return {r["resource_id"]: dict(r) for r in rows}


def prerequisite_order(skills: list[str], known: set[str]) -> list[str]:
    """Return a deterministic prerequisite-respecting competency sequence for any goal."""
    by_skill = {r["skill"]: r for r in RESOURCES}
    remaining = list(dict.fromkeys(skills))
    ordered: list[str] = []
    satisfied = {x.lower() for x in known}

    while remaining:
        available = []
        for idx, skill in enumerate(remaining):
            r = by_skill.get(skill, {})
            prereqs = [str(x).lower() for x in r.get("prerequisites", [])]
            if all(p in satisfied for p in prereqs):
                available.append((idx, skill))
        if not available:
            # Broken/missing prerequisite data should not deadlock the roadmap.
            available = [(0, remaining[0])]
        _, chosen = min(available)
        ordered.append(chosen)
        remaining.remove(chosen)
        satisfied.add(chosen.lower())
    return ordered


def preference_score(r: dict[str, Any], preference: str) -> float:
    p = normalize(preference)
    tags = {normalize(x) for x in r["tags"]}
    if p in tags:
        return 1.0
    if p == "project focused" and r["type"].lower() == "project":
        return 1.0
    if p == "short sessions" and r["hours"] <= 8:
        return 1.0
    if p == "hands on" and (r["type"] == "Project" or "project" in tags):
        return 1.0
    return 0.25


def _match_text_terms(text: str, terms: list[str]) -> float:
    t = normalize(text)
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if re.search(r"(?<![a-z0-9])" + re.escape(normalize(term)) + r"(?![a-z0-9])", t))
    return hits / len(terms)


def _skill_level_map(p: Profile) -> dict[str, int]:
    """Infer explicit proficiency from profile phrases; default known skills to intermediate."""
    result = {}
    for raw in p.skills + p.completed:
        skillset = extract_skills(raw)
        text = normalize(raw)
        level = 2
        if any(x in text for x in ["beginner", "basic", "basics", "novice"]): level = 1
        elif any(x in text for x in ["advanced", "expert", "strong"]): level = 3
        for skill in skillset:
            result[skill] = max(result.get(skill, 0), level)
    return result


def recommend_resources(p: Profile) -> dict[str, Any]:
    goal = find_goal(p.goal) or p.goal
    if goal not in GOALS:
        goal = "Backend Developer"
    required = goal_requirements(goal)
    skill_levels = _skill_level_map(p)
    known = set(skill_levels)
    pm = progress_map(p.learner_id)
    feedback = feedback_map(p.learner_id)
    completed_resource_ids = {rid for rid, x in pm.items() if x["status"] == "completed"}
    # Accept resource IDs in the profile's completed list as well as skill names.
    completed_resource_ids.update(x for x in p.completed if any(r["id"] == x for r in RESOURCES))

    query_parts = [goal, p.experience, p.preference, *p.interests]
    for skill in p.skills + p.completed:
        query_parts.append(skill)
    query = " ".join(query_parts)
    semantic = cosine_similarity(VECTORIZER.transform([query]), RESOURCE_MATRIX)[0]

    missing = [s for s in required if s not in known]
    ordered_skills = prerequisite_order(missing, known)
    priority = {skill: 1.0 - (i / max(len(ordered_skills), 1)) for i, skill in enumerate(ordered_skills)}
    interest_terms = [normalize(i) for i in p.interests]
    profile_text = " ".join([goal, p.experience, *p.interests, *p.skills])

    candidates = []
    for idx, r in enumerate(RESOURCES):
        if r["skill"] not in required or r["id"] in completed_resource_ids:
            continue
        # Never recommend an already-mastered competency unless the user is asking for reinforcement.
        if r["skill"] in skill_levels and skill_levels[r["skill"]] >= LEVELS.get(r["level"], 2):
            continue
        level_delta = abs(LEVELS.get(r["level"], 2) - LEVELS.get(p.experience, 1))
        level_match = 1.0 - min(level_delta, 2) / 2
        tag_match = _match_text_terms(" ".join(r["tags"]), interest_terms)
        title_match = _match_text_terms(r["title"] + " " + r["skill"], interest_terms)
        interest = max(tag_match, title_match)
        rating = r["rating"] / 5
        semantic_score = float(semantic[idx])
        pref = preference_score(r, p.preference)
        type_fit = 1.0 if r["type"] == "Course" else (1.0 if normalize(p.preference) in {"project focused", "hands on"} else 0.55)
        feedback_signal = max(-0.12, min(0.12, 0.025 * feedback.get(r["id"], 0)))
        # Prerequisite readiness: resources whose prerequisites are known are boosted; blocked items are deferred.
        prereqs = {x.lower() for x in r.get("prerequisites", [])}
        ready = prereqs.issubset({x.lower() for x in known})
        prerequisite_fit = 1.0 if ready else 0.35
        skill_priority = priority.get(r["skill"], 0.0)
        score = (
            0.30 * skill_priority +
            0.22 * semantic_score +
            0.14 * prerequisite_fit +
            0.10 * level_match +
            0.08 * interest +
            0.06 * rating +
            0.05 * pref +
            0.05 * type_fit +
            0.05 * min(1.0, len(r.get("tags", [])) / 5) +
            feedback_signal
        )
        candidates.append((r, score, semantic_score, level_match, interest, pref, ready))

    # Pick one best resource per competency in prerequisite order. This prevents duplicate alternatives
    # from crowding the actual learning path.
    ranked = []
    for skill in ordered_skills:
        group = [x for x in candidates if x[0]["skill"] == skill]
        group.sort(key=lambda x: x[1], reverse=True)
        if group:
            ranked.append(group[0])

    # Add a small number of alternatives, but never ahead of the primary competency sequence.
    selected_ids = {x[0]["id"] for x in ranked}
    alternatives = sorted((x for x in candidates if x[0]["id"] not in selected_ids), key=lambda x: x[1], reverse=True)
    ranked.extend(alternatives[:4])

    resources = []
    for r, score, sem, level, interest, pref, ready in ranked:
        resources.append({
            **r,
            "match_score": round(max(0.01, min(score, 0.99)), 3),
            "reason": explain(r, p, goal, sem, level, interest, pref, feedback.get(r["id"], 0), ready),
            "status": pm.get(r["id"], {}).get("status", "not_started"),
            "assessment_score": pm.get(r["id"], {}).get("score"),
        })

    total_hours = sum(r["hours"] for r in resources if r["status"] != "completed")
    weeks = max(1, math.ceil(total_hours / max(p.hours_per_week, 1))) if resources else 0
    milestones = [{"order": i, "title": r["title"], "skill": r["skill"], "type": r["type"], "status": r["status"]} for i, r in enumerate(resources, 1)]

    mastery = {skill: (skill_levels.get(skill, 0) / 3 * 100) for skill in required}
    for r in resources:
        if r["assessment_score"] is not None:
            mastery[r["skill"]] = r["assessment_score"]
    completed_count = sum(1 for r in resources if r["status"] == "completed")
    progress = round(100 * completed_count / max(len(resources), 1))

    return {
        "goal": goal, "known_skills": sorted(known), "missing_skills": ordered_skills,
        "resources": resources, "estimated_weeks": weeks, "total_hours": total_hours,
        "progress": progress, "mastery": mastery, "milestones": milestones,
        "method": "Hybrid semantic + prerequisite readiness + proficiency + profile + feedback scoring",
    }


def explain(r: dict[str, Any], p: Profile, goal: str, sem: float, level: float, interest: float, pref: float, feedback_signal: int = 0, ready: bool = True) -> str:
    parts = [f"{r['skill']} is part of the {goal} competency map"]
    if r["prerequisites"]:
        parts.append("its prerequisites fit the recommended sequence" if ready else "its prerequisites will be covered before this step")
    if interest:
        parts.append("it matches one of your interests")
    if level > 0.5:
        parts.append(f"its {r['level'].lower()} level fits your experience")
    if pref >= 0.9:
        parts.append("it matches your preferred learning style")
    if feedback_signal > 0:
        parts.append("your earlier feedback favored this resource")
    elif feedback_signal < 0:
        parts.append("its rank is reduced by earlier skip/not-helpful feedback")
    return "; ".join(parts) + "."


# ---------- API ----------
@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "resources": len(RESOURCES), "ai_mode": "LLM-enhanced" if os.getenv("OPENAI_API_KEY") else "Local hybrid recommender"}


@app.get("/api/resources")
def resources():
    return {"resources": RESOURCES}


@app.post("/api/profile/analyze")
def analyze_profile(profile: Profile):
    return {"goal": find_goal(profile.goal) or profile.goal, "skills": sorted(canonical_skills(profile.skills + profile.completed)), "experience": profile.experience, "skill_levels": _skill_level_map(profile)}


@app.post("/api/recommend")
def recommend(profile: Profile):
    return recommend_resources(profile)


@app.post("/api/progress")
def update_progress(req: ProgressRequest):
    with db() as conn:
        conn.execute("INSERT INTO progress(learner_id,resource_id,status,score,updated_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(learner_id,resource_id) DO UPDATE SET status=excluded.status, score=COALESCE(excluded.score,progress.score), updated_at=CURRENT_TIMESTAMP", (req.learner_id, req.resource_id, req.status, req.score))
    return {"ok": True, "progress": progress_map(req.learner_id).get(req.resource_id)}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    with db() as conn:
        conn.execute("INSERT INTO feedback(learner_id,resource_id,action) VALUES(?,?,?)", (req.learner_id, req.resource_id, req.action))
    return {"ok": True, "message": "Feedback recorded. Your next recommendation can use this signal."}


@app.post("/api/assessment")
def assessment(req: AssessmentRequest):
    status = "completed" if req.score >= 70 else "in_progress"
    with db() as conn:
        conn.execute("INSERT INTO progress(learner_id,resource_id,status,score,updated_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(learner_id,resource_id) DO UPDATE SET status=excluded.status, score=excluded.score, updated_at=CURRENT_TIMESTAMP", (req.learner_id, req.resource_id, status, req.score))
    return {"ok": True, "status": status, "message": "Strong mastery — you can move forward." if req.score >= 70 else "Let's reinforce this skill before moving on."}


@app.post("/api/chat")
def chat(req: ChatRequest):
    text = req.message.strip()
    t = normalize(text)
    detected_goal = find_goal(text)
    detected_skills = extract_skills(text)
    p = req.profile.model_copy()
    if detected_goal:
        p.goal = detected_goal
    if detected_skills:
        p.skills = sorted(canonical_skills(p.skills + detected_skills))
    path = recommend_resources(p)

    if os.getenv("OPENAI_API_KEY") and os.getenv("PATHPILOT_LLM_ENABLED", "true").lower() == "true":
        llm = call_llm(text, p, path)
        if llm:
            return {"reply": llm, "profile_updates": {"goal": p.goal, "skills": p.skills}}

    if any(x in t for x in ["why", "recommend"]):
        first = path["resources"][0] if path["resources"] else None
        reply = first["reason"] if first else "Your selected goal has no remaining gaps in the current catalog."
    elif "skill" in t or "gap" in t:
        reply = f"You currently have {len(path['known_skills'])} detected skills. Priority gaps are: {', '.join(path['missing_skills'][:6]) or 'none'}."
    elif "time" in t or "week" in t or "how long" in t:
        reply = f"At {p.hours_per_week} hours/week, the remaining roadmap is about {path['estimated_weeks']} weeks ({path['total_hours']} learning hours)."
    elif "next" in t or "start" in t or "roadmap" in t or "path" in t or detected_goal:
        first = path["resources"][0] if path["resources"] else None
        reply = f"For {path['goal']}, I found {len(path['missing_skills'])} priority gaps. Start with {first['title']} — it builds {first['skill']} before the next prerequisite." if first else f"Your {path['goal']} profile is already well covered."
    else:
        reply = "I can analyze your goal, identify skill gaps, explain recommendations, estimate your timeline, or tell you what to learn next."
    return {"reply": reply, "profile_updates": {"goal": p.goal, "skills": p.skills}}


def call_llm(message: str, profile: Profile, path: dict[str, Any]) -> str | None:
    """Optional LLM layer. The product remains fully functional without an API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    endpoint = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = f"""You are PathPilot, a concise learning strategist. Answer the learner using only this structured plan.\nGoal: {path['goal']}\nKnown skills: {path['known_skills']}\nGaps: {path['missing_skills']}\nNext resources: {[r['title'] for r in path['resources'][:5]]}\nLearner message: {message}\nExplain recommendations concretely and do not invent courses."""
    payload = json.dumps({"model": model, "input": prompt, "max_output_tokens": 300}).encode()
    try:
        req = Request(endpoint, data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
        if data.get("output_text"):
            return data["output_text"].strip()
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("text"):
                    return content["text"].strip()
    except Exception:
        return None
    return None
