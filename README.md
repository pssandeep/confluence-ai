# Confluence Cloud Agentic AI

Building an Agentic AI support assistant for Atlassian Confluence Cloud.

## What this builds
- LangGraph multi-agent pipeline (triage → KB search → resolution)
- RAG pipeline over real Confluence Cloud content (ChromaDB + nomic-embed-text)
- MCP server wrapping the Confluence Cloud REST API
- FastAPI streaming backend + React Chat UI

## Stack
Python 3.11 · LangChain 0.2 · LangGraph · ChromaDB · Ollama + LLaMA 3 · FastAPI · React 18

## Setup
See INSTRUCTIONS.md for full setup guide.

> ⚠️ Never commit your `.env` file. API keys and Confluence credentials stay local.
