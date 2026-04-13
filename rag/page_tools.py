from langchain_ollama import OllamaLLM
import json, re, os
from dotenv import load_dotenv

load_dotenv()

# Why underscore prefix on _llm: signals "private to this module — don't import directly"
# Why os.getenv with fallback: reads LOCAL_LLM_MODEL from .env if set,
# defaults to llama3. On Day 9 you swap to Claude by changing one line in .env.
_llm = OllamaLLM(model=os.getenv("LOCAL_LLM_MODEL", "llama3"), temperature=0)


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and prose prefixes, then parse JSON."""
    # Step 1: strip backtick fences
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Step 2: find where JSON actually starts — skips any "Here is the JSON:" prefix
    json_start = clean.find("{")
    if json_start == -1:
        json_start = clean.find("[")
    if json_start != -1:
        clean = clean[json_start:]
    # Step 3: trim anything after the last closing brace
    json_end = max(clean.rfind("}"), clean.rfind("]"))
    if json_end != -1:
        clean = clean[:json_end + 1]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw_response": clean, "parse_error": True}


def summarise_page(page_title: str, page_content: str) -> dict:
    """Summarise a Confluence Cloud page and extract key metadata."""
    truncated = page_content[:3000]
    if len(page_content) > 3000:
        truncated += "\n[... content truncated ...]"
    prompt = f"""You are a Confluence Cloud KB assistant.

Analyse this page and return ONLY JSON with:
- "summary": 2-3 sentence plain-English summary
- "key_topics": list of up to 5 main topics covered
- "is_troubleshooting_guide": true or false
- "related_components": Confluence Cloud components mentioned
- "target_audience": one of [Admin, Developer, End User, All]

Title: {page_title}
Content:
{truncated}

Return ONLY valid JSON. Do not include any text before or after the JSON."""
    return _parse_json(_llm.invoke(prompt))


def extract_issue_details(ticket_text: str) -> dict:
    """Extract structured info from a Confluence Cloud support ticket."""
    prompt = f"""You are a Confluence Cloud support triage system.

Analyse this support ticket and return ONLY a JSON object with these exact keys:
- "category": one of [Performance, Authentication/SSO, Export/Import, Macro, Permissions, Email, Search, Attachments, Billing/License, Integration, Other]
- "severity": one of [Critical, High, Medium, Low]
- "affected_component": the specific Confluence Cloud component or feature
- "is_cloud_specific": true if this issue is specific to Confluence Cloud
- "suggested_first_steps": list of 3 strings — first things to investigate
- "needs_escalation": true if platform bug, false if config issue

Return ONLY valid JSON. Do not include any text before or after the JSON.

Ticket:
{ticket_text}"""
    return _parse_json(_llm.invoke(prompt))


if __name__ == "__main__":
    print("=== Testing extract_issue_details ===\n")
    result = extract_issue_details(
        "Confluence Cloud is slow for all users after our SSO config change this morning."
    )
    print(json.dumps(result, indent=2))

    print("\n=== Testing summarise_page ===\n")
    result = summarise_page(
        "How to configure SSO in Confluence Cloud",
        """This guide covers setting up Single Sign-On (SSO) for Confluence Cloud.
        SSO allows users to log in once and access multiple Atlassian products.
        ## Prerequisites
        You need admin access to both Confluence Cloud and your identity provider (Okta, Azure AD).
        ## Steps
        1. Go to Admin > Security > SAML single sign-on
        2. Enter your Identity Provider metadata URL
        3. Map the email attribute from your IdP to the Atlassian account email
        4. Test with a single user before enabling for all users."""
    )
    print(json.dumps(result, indent=2))
