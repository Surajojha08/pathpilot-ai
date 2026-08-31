# PathPilot AI — Personalized Learning Path Recommender

PathPilot is an AI-powered learning assistant prototype that converts a learner's natural-language goal and existing skills into a personalized, prerequisite-aware learning roadmap. Its recommender uses a hybrid scoring pipeline with semantic similarity, prerequisite readiness, proficiency, experience, interests, learning preference, resource quality, and learner feedback.

## Features
- Conversational learning assistant
- Learner profile and skill-gap analysis
- Goal-specific prerequisite graph/order
- Course, project and learning-resource recommendations
- Explainable recommendations (“Why this?”)
- Adaptive roadmap based on current skills and learning capacity
- Progress-oriented dashboard
- Responsive dark UI
- Works without an external AI API; an optional LLM layer can enrich natural-language responses while the structured recommender remains the source of truth
- Explicit skill proficiency detection and mastery-aware filtering
- Offline evaluation with Precision@3, Recall@3 and NDCG@3 across 18 learner scenarios
- Automated unit tests for goal extraction, prerequisite ordering and personalization

## Architecture
`Browser UI → FastAPI → NLP/Profile Engine → Skill Gap + Proficiency Engine → Semantic Retrieval → Hybrid Ranking → Prerequisite-aware Path Generator → Progress/Feedback Loop`

## Run locally
Requirements: Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Demo flow
1. Open AI Assistant.
2. Enter: `I want to become a backend developer. I know Java and basic SQL.`
3. PathPilot extracts the goal/skills and builds a pathway.
4. Open Learning Path to see ordered resources.
5. Ask `Why is this recommended?`, `What are my skill gaps?`, or `How long will it take?`.

## Project structure
- `app/main.py` — API, learner profiling, skill-gap and recommendation logic
- `data/resources.json` — learning resource catalog
- `static/` — responsive dashboard and conversational UI
- `docs/` — submission documentation

## Evaluation
Run:

```bash
PYTHONPATH=. python evaluate.py
PYTHONPATH=. pytest -q
```

The benchmark is intentionally described as an internal validation set. Do not claim that its score represents real-world accuracy. Production evaluation should use held-out learner interactions and expert relevance labels.

## Extension roadmap
For production, connect a vector database for semantic resource retrieval, an LLM for richer profile extraction/explanations, authentication, persistent learner progress, assessment scoring, and real-time course catalogs.
