# Multimodal RAG & Multi-Agent Recommendation System



---

## Project Overview

a 3-module course building a complete 
**Connoisseur Companion** restaurant recommendation system from scratch:

- **Module 1** — LLM-powered structured data extraction + multimodal processing
- **Module 2** — Multimodal RAG with ChromaDB vector indexes + fusion ranking
- **Module 3** — 6 specialized agents + LangGraph workflow + Gradio chatbot

Applied across **7 hands-on notebooks** using **GPT-5**, **LangChain**, 
**LangGraph**, and **ChromaDB** on the California Culinary dataset.

**Domain:** Multimodal RAG + Multi-Agent AI Systems  
**LLMs:** GPT-5 (OpenAI) + IBM Granite  
**Framework:** LangChain + LangGraph  
**Dataset:** California Culinary Map (100+ restaurants)

---

## 📂 7 Notebooks — 3 Modules

```
AI0351EN/
│
├── M1 — Data Engineering
│   ├── Structure Text and Multimodal Data with LLMs.ipynb
│   └── M1L2_Process_Multimodal_Data_with_LLMs.ipynb
│
├── M2 — Multimodal RAG
│   ├── M2L1_Lab.ipynb   — Construct Multimodal Vector Index
│   ├── M2L2_Lab.ipynb   — Similarity Retrieval + Metadata Filtering
│   └── M2L3_Lab.ipynb   — Similarity Fusion + Retrieval Ranking
│
├── M3 — Multi-Agent System
│   ├── M3L1_Design_Specialized_Agents.ipynb
│   ├── M3L2_Implement_Multi_Agent_Systems.ipynb
│   └── M3L3_Build_Chatbot_Interface.ipynb
│
└── California-Culinary-Map.txt   # 100+ restaurant knowledge base
```

---

##  Module 1 — LLM Data Engineering

### Lab 1 — Structure Unstructured Restaurant Data
```python
# IBM Granite 3 8B → extract 7 fields from raw text paragraphs
restaurant_data_structure_prompt_generation(paragraph)
# → name, location, cuisine, style, rating, description, price_range
# + JSON auto-repair loop if LLM output is malformed
```

### Lab 2 — Process Multimodal Data with LLMs
- Load food recipe + user visit data from JSON
- Process text + image data through LLM pipelines
- Build augmented multimodal knowledge base from California Culinary Map

---

## 🔍 Module 2 — Multimodal RAG Pipeline

### M2L1 — Construct Multimodal Vector Index
```python
# Build ChromaDB collections for text AND images
text_collection = chroma_client.create_collection("restaurant_articles")
image_collection = chroma_client.create_collection("food_images")
# Embed text chunks + image features → store with metadata
```

### M2L2 — Similarity Retrieval + Metadata Filtering
```python
# Top-K semantic retrieval with filters
results = collection.query(
    query_texts=[user_query],
    n_results=5,
    where={"cuisine": "Italian", "price_range": "$$"}
)
```

### M2L3 — Similarity Fusion + Retrieval Ranking
```python
# Cosine similarity scores from both text + image indexes
text_scores = compute_cosine_similarity(query_emb, text_embs)
image_scores = compute_cosine_similarity(query_emb, image_embs)

# Fuse scores → unified ranked list
fused_ranking = fusion_rank(text_scores, image_scores, weights=[0.6, 0.4])
```

---

## 🤖 Module 3 — Multi-Agent Recommendation System

### M3L1 — Design 6 Specialized Agents (LangChain + GPT-5)

| # | Agent | Role |
|---|---|---|
| 1 | **User Profile Generator** | Analyze visit history → extract preferences & dining patterns |
| 2 | **Semantic Retrieval Agent** | Query multimodal vector DB → retrieve matching restaurants |
| 3 | **Restaurant Recommender** | Match restaurants to user profile |
| 4 | **Recipe Suggester** | Suggest recipes based on cuisine preferences |
| 5 | **Review Analyzer** | Sentiment analysis on user reviews |
| 6 | **Response Synthesizer** | Combine all outputs → final natural language recommendation |

Each agent defined with **role**, **goal**, and **backstory** for specialized expertise.

### M3L2 — Implement Multi-Agent Workflow (LangGraph + GPT-5)
```python
# LangGraph TypedDict state
class RecommendationState(TypedDict):
    user_data: Dict
    user_profile: Dict
    retrieved_restaurants: List
    recommendations: List
    recipes: List
    sentiment_analysis: Dict
    final_response: str

# Parallel execution with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(agent_fn, state) for agent_fn in parallel_agents]

# LangGraph hybrid workflow — sequential + parallel nodes
graph.add_edge("user_profile", "retrieval")
graph.add_edge("retrieval", ["recommender", "recipe_suggester"])  # Parallel fan-out
graph.add_edge(["recommender", "recipe_suggester"], "review_analyzer")
graph.add_edge("review_analyzer", "synthesizer")
```

### M3L3 — Gradio Chatbot Interface
```python
# Multi-tab Gradio chatbot
# Tab 1: Single-turn Q&A
# Tab 2: Multi-turn conversation with history
# Tab 3: Full recommendation pipeline UI
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLMs | GPT-5 (OpenAI) + IBM Granite |
| Frameworks | LangChain 0.3.0 + LangGraph 0.2.0 |
| Vector DB | ChromaDB (text + image collections) |
| Multimodal | Text + image embeddings |
| Retrieval | Cosine similarity + metadata filtering + fusion ranking |
| Parallel exec | `ThreadPoolExecutor` |
| UI | Gradio (multi-tab chatbot) |

---

## 🎓 Skills Demonstrated

- LLM text-to-JSON extraction with auto-repair loop
- Multimodal ChromaDB indexing (text + images)
- Semantic similarity retrieval + metadata filtering
- Cosine similarity fusion ranking across modalities
- 6 specialized agent design (role + goal + backstory)
- LangGraph hybrid workflow — sequential + parallel execution
- ThreadPoolExecutor for parallel agent calls
- Multi-tab Gradio chatbot interface
- End-to-end RAG + multi-agent recommendation pipeline
- California Culinary Map as real-world domain dataset

---

