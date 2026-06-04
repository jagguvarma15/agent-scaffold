# Framework-Mixed Aliases

A Python recipe whose prose mentions BOTH LangGraph and Pydantic AI as
candidate frameworks. SR2 filtering should load only the matching one when a
framework is selected, but load both when framework is "none".

The recipe relies on pattern: RAG and uses LangGraph for state graphs.
The Pydantic AI variant is also viable for simpler cases.
