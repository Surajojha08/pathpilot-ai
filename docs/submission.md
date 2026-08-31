# PathPilot AI — Submission Notes

## One-line pitch
PathPilot AI converts a learner's natural-language career goal into an adaptive, explainable and prerequisite-aware learning roadmap.

## Problem addressed
Online learning platforms provide many resources but often leave learners to decide what to learn first, what prerequisites they are missing and when they are ready to move forward.

## Solution
PathPilot combines a learner model, goal competency map, skill-gap analysis, semantic resource matching and prerequisite-aware sequencing. It then adapts the roadmap using completion, assessment and feedback signals.

## AI / intelligent components
1. Natural-language goal and skill extraction.
2. TF-IDF semantic relevance between learner profile and resources.
3. Multi-factor recommendation scoring.
4. Prerequisite graph ordering.
5. Adaptive progress/mastery updates.
6. Optional LLM layer for richer conversational explanations.

## Why it is more than a chatbot
The chatbot is an interface to a structured recommendation engine. The roadmap is generated from explicit competency requirements, prerequisites and learner-state signals. This makes the recommendation reproducible and explainable.

## Example
Learner: `I want to become a backend developer. I know Java and basic SQL.`

PathPilot identifies REST APIs, Spring Boot, JPA/Hibernate, Docker, System Design and Backend Engineering as remaining target skills and orders the resources around prerequisites.

## Evaluation statement
The repository contains `evaluate.py`, which measures Precision@3, Recall@3 and NDCG@3 on six curated demo scenarios. These metrics validate the prototype's ranking behavior, not general population accuracy. A real deployment should use a held-out dataset of learner interactions and expert relevance labels.

## Technology
- Frontend: HTML/CSS/JavaScript
- Backend: Python/FastAPI
- Recommendation: scikit-learn TF-IDF + cosine similarity + interpretable weighted ranking
- Persistence: SQLite
- Optional AI assistant: external LLM API via environment configuration

## Demo checklist
- [ ] Start the FastAPI server.
- [ ] Show natural-language goal input.
- [ ] Show profile and skill-gap extraction.
- [ ] Show prerequisite-aware roadmap.
- [ ] Show recommendation explanation and match score.
- [ ] Mark an item complete.
- [ ] Submit an assessment score.
- [ ] Give helpful/skip feedback.
- [ ] Refresh the roadmap and show changed progress/mastery.
