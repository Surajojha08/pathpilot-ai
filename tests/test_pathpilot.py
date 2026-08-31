from app.main import Profile, extract_skills, find_goal, prerequisite_order, recommend_resources


def test_goal_and_skill_extraction():
    text = "I know Java, SQL and Spring Boot and want to become a backend developer"
    assert find_goal(text) == "Backend Developer"
    assert {"Java", "SQL", "Spring Boot"}.issubset(set(extract_skills(text)))


def test_prerequisites_are_ordered():
    order = prerequisite_order(["SQL", "REST APIs", "Spring Boot", "JPA/Hibernate", "Docker", "System Design"], {"Java"})
    assert order.index("SQL") < order.index("REST APIs") < order.index("Spring Boot")
    assert order.index("Docker") < order.index("System Design")


def test_backend_path_personalizes_known_skills():
    out = recommend_resources(Profile(goal="Backend Developer", experience="Intermediate", skills=["Java", "SQL"], interests=["backend"], hours_per_week=10))
    assert out["missing_skills"][:3] == ["REST APIs", "Spring Boot", "JPA/Hibernate"]
    assert out["resources"][0]["id"] == "rest-api"
    assert out["resources"][0]["match_score"] > 0


def test_ai_path_starts_with_python_for_beginner():
    out = recommend_resources(Profile(goal="AI/ML Engineer", experience="Beginner", skills=[], interests=["AI"], hours_per_week=7))
    assert out["resources"][0]["skill"] == "Python"
