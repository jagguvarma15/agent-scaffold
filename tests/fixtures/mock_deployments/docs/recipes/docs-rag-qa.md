# Docs RAG QA

A simple Q&A agent that retrieves from your own documentation.

The recipe relies on pattern: RAG and uses Qdrant for the vector store. The
TypeScript variant prefers the Vercel AI SDK while the Python variant uses
Pydantic AI.

We also need rate limiting on user queries.
