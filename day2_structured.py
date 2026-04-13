from langchain_ollama import OllamaLLM
import json, re

llm = OllamaLLM(model="llama3", temperature=0)

def extract_issue_details(ticket_text: str) -> dict:
    """Extract structured info from a Confluence Cloud support ticket."""

    prompt = f"""You are a Confluence Cloud support triage system.

Analyse this support ticket and return ONLY a JSON object with these exact keys:
- "category": one of [Performance, Authentication/SSO, Export/Import, Macro, Permissions, Email, Search, Attachments, Billing/License, Integration, Other]
- "severity": one of [Critical, High, Medium, Low]
- "affected_component": the specific Confluence Cloud component or feature
- "is_cloud_specific": true if this issue is specific to Confluence Cloud
- "suggested_first_steps": list of 3 strings — first things to investigate
- "needs_escalation": true if this looks like a platform bug, false if config issue

Return ONLY valid JSON. No prose. No markdown code fences.

Ticket:
{ticket_text}"""

    raw = llm.invoke(prompt)

    # Why re.sub: LLMs sometimes wrap JSON in ```json ... ``` even when told not to.
    # This strips those backticks before we try to parse.
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Why try/except: if the model still produces malformed JSON, return a
    # safe fallback dict instead of crashing the whole program.
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw_response": clean, "parse_error": True}


tickets = [
    """
    Subject: Space export 100% complete but no download link

    Hi team, we exported our Product space in Confluence Cloud. The export job
    shows 100% complete in the notification bell, but when we click it there is
    no download link — just a spinner. We have tried this 3 times over 2 days
    and the same thing happens. The space has about 1,800 pages.
    """,
    """
    Subject: CRITICAL — All Okta SSO users cannot log in to Confluence

    Since 9 AM this morning, none of our users can authenticate via Okta.
    They are redirected back to the login page with no error message.
    Admin accounts using Atlassian credentials still work. We changed
    nothing on our end. We have 450 active users blocked from production.
    """,
]

print("=== Structured Issue Extraction ===\n")
for i, ticket in enumerate(tickets, 1):
    result = extract_issue_details(ticket)
    print(f"Ticket {i}:")
    print(json.dumps(result, indent=2))
    print()
    