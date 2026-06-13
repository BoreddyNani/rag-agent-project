---
title: RAG Document Chat
emoji: 📄
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
python-version: 3.12
---

# RAG Document Chat

Upload any PDF and ask questions about it.
Built with LangChain + ChromaDB + Gemini 2.0 Flash.
Hybrid retrieval: BM25 + dense vector search with RRF fusion.
Evaluated with RAGAS — faithfulness: [X], context recall: [Y].