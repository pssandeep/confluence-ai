# ~/confluence-ai/day1_classifier.py

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3", temperature=0)

ISSUE_CATEGORIES = [
    "Performance / slowness",
    "Authentication / SSO login",
    "Space export / import",
    "Macro not working",
    "Permission / access denied",
    "Email / notification",
    "Search not returning results",
    "Attachment upload failure",
    "Billing / license / admin",
    "Integration (Jira, Slack, etc.)",
]

def classify_confluence_issue(issue_description: str) -> str:
    categories_list = "\n".join(f"- {c}" for c in ISSUE_CATEGORIES)
    prompt = f"""You are a Confluence Cloud support triage assistant.

Classify the following issue into exactly ONE of these categories:
{categories_list}

Issue: {issue_description}

Reply with only the category name. Nothing else."""
    return llm.invoke(prompt).strip()


test_issues = [
    "Users cannot log in via Okta SSO — they get 'User not found in directory' error.",
    "The space export completed at 100% but there is no download button — it just shows a spinner.",
    "The Jira Issues macro shows 'Failed to load' even though our Jira Cloud is connected.",
    "Confluence Cloud dashboard takes 40 seconds to load for users in the Singapore office.",
]

print("=== Confluence Cloud Issue Classifier ===\n")
for issue in test_issues:
    category = classify_confluence_issue(issue)
    print(f"Issue:    {issue[:75]}...")
    print(f"Category: {category}\n")


# ── Step 7: Temperature demo ──────────────────────────────────────────────────
print("\n=== Temperature demo ===")
issue = "Users are saying Confluence Cloud is slow"

for temp in [0.0, 0.5, 1.0]:
    llm_t = OllamaLLM(model="llama3", temperature=temp)
    response = llm_t.invoke(f"In one sentence, what should I check first for: {issue}")
    print(f"\ntemperature={temp}: {response}")