from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = OllamaLLM(model="llama3", temperature=0)

system = """You are a Confluence Cloud support specialist.
Help the user diagnose their Confluence Cloud issue step by step.
Ask one clarifying question at a time. Be concise."""


def chat_turn(history: list, user_message: str) -> str:
    # Add user message to history BEFORE calling the LLM
    history.append(HumanMessage(content=user_message))

    # Why MessagesPlaceholder: injects the full history list into the prompt
    # at the "history" slot. Every call gets the complete conversation so far.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
    ])

    chain = prompt | llm
    response = chain.invoke({"history": history})

    # Add the model's response to history so the NEXT turn sees it
    history.append(AIMessage(content=response))
    return response


# ── Simulate a real support conversation ─────────────────────────────────────

history = []  # starts empty — grows with every turn

turns = [
    "My space export is not working.",
    "Confluence Cloud. The space has about 1,800 pages.",
    "The progress bar says 100% complete but there is no download link.",
    "I checked my email and there is nothing there either.",
]

print("=== Multi-turn Support Conversation ===\n")
for turn in turns:
    print(f"User:  {turn}")
    response = chat_turn(history, turn)
    print(f"Agent: {response}\n")