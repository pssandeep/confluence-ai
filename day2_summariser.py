from langchain_ollama import OllamaLLM
import json, re

llm = OllamaLLM(model="llama3", temperature=0)


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and any prose prefix, then parse JSON."""

    # Step 1: strip backtick fences
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Step 2: find where the JSON actually starts
    # Why: models sometimes add "Here is the JSON:" before the actual object.
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

    raw = llm.invoke(prompt)
    return _parse_json(raw)


# ── Test data ─────────────────────────────────────────────────────────────────

sample_page = {
    "title": "Troubleshooting Confluence Cloud Space Export Issues",
    "content": """
Space exports in Confluence Cloud can fail or produce incomplete results. This guide
covers the most common causes seen in Cloud environments.

## Export Shows 100% But No Download Link

This is a known intermittent issue in Confluence Cloud. The export job completes
server-side but the notification system fails to deliver the download link.

Resolution:
1. Wait 10 minutes and check your email — Atlassian sends an email with the download
   link as a fallback when the in-app notification fails.
2. Check Atlassian Status (status.atlassian.com) for any ongoing incidents in your region.
3. If no email arrives after 30 minutes, raise a support ticket with Atlassian.
   Include: your Cloud site name, space key, and the exact time you triggered the export.

## Export Times Out on Large Spaces

Confluence Cloud has a 2-hour timeout for space exports. Spaces over 2GB may hit this.

Resolution:
1. Split the export: export by date range rather than the full space.
2. Use the Confluence Cloud REST API to export pages in batches programmatically.
    """
}

result = summarise_page(sample_page["title"], sample_page["content"])
print("Page Summary:")
print(json.dumps(result, indent=2))