from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="llama3", temperature=0)

# ── Without a system prompt ───────────────────────────────────────────────────
bad_response = llm.invoke("What causes Confluence export to fail?")
print("WITHOUT system prompt:")
print(bad_response[:300])

# ── With a focused system prompt ──────────────────────────────────────────────
# Why ChatPromptTemplate: separates the system instruction from the user message.
# Why the pipe |: LangChain chain syntax — prompt_template | llm means
# "run the template first, feed its output into the LLM".
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a Confluence Cloud Level 2 support specialist.
You only answer questions about Atlassian Confluence Cloud (not Server or DC).
Keep answers concise — bullet points preferred. Maximum 5 bullets per answer.
If a question is not about Confluence Cloud, say: 'Outside my Confluence Cloud scope.'"""),
    ("human", "{question}"),
])

chain = prompt_template | llm
better_response = chain.invoke({
    "question": "What causes a Confluence Cloud space export to fail silently?"
})
print("\nWITH system prompt:")
print(better_response[:500])