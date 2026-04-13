from langchain_ollama import OllamaLLM

# Why: OllamaLLM wraps the local Ollama server so we can call LLaMA 3
# with the same interface we'll later swap for Claude or GPT-4o on Day 9.
llm = OllamaLLM(model="llama3", temperature=0)

# Real Confluence Cloud support question
question = (
    "A user reports that their Confluence Cloud space export is stuck. "
    "The progress bar reached 100% but no download link appeared. "
    "What are the most likely causes?"
)

print("Question:", question)
print("\nAnswer:")
print(llm.invoke(question))



from langchain_ollama import OllamaEmbeddings

embedder = OllamaEmbeddings(model="nomic-embed-text")

# Why nomic-embed-text: purpose-built for document retrieval.
# It understands technical language better than general embeddings.

confluence_titles = [
    "Confluence Cloud space export stuck at 100 percent with no download",
    "Export hangs and download never appears in Confluence Cloud",
    "How to configure Confluence Cloud email notifications",
    "Setting up SSO with Okta in Confluence Cloud",
]

embeddings = embedder.embed_documents(confluence_titles)

print(f"\nEach embedding has {len(embeddings[0])} dimensions")
print(f"First 5 values of title 1: {[round(x, 4) for x in embeddings[0][:5]]}")


import numpy as np

def cosine_similarity(v1, v2):
    # Why cosine: measures the angle between two meaning-vectors.
    # 1.0 = identical meaning · 0.0 = unrelated · -1.0 = opposite meaning
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

print("\n--- Similarity scores ---")
print(f"Export stuck vs Export hangs:        {cosine_similarity(embeddings[0], embeddings[1]):.3f}  ← should be HIGH")
print(f"Export stuck vs Email notifications: {cosine_similarity(embeddings[0], embeddings[2]):.3f}  ← should be LOW")
print(f"Export stuck vs SSO setup:           {cosine_similarity(embeddings[0], embeddings[3]):.3f}  ← should be LOW")
