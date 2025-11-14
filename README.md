# Bonus 1 — Design Notes

This project required answering natural-language questions about member messages.  
Below are the alternative approaches I evaluated while designing the solution.

---

## 1. Pure Rule-Based Extraction (Regex + Pattern Matching)
**Idea:** Hand-write intent patterns such as:
- “When is X planning a trip to Y?”
- “How many cars does X have?”
- “What are X's favorite restaurants?”

**Pros**
- Very fast and deterministic.
- No additional infrastructure (vector search, models, etc.).
- Easy to debug.

**Cons**
- Fragile to rephrasing (“headed to Paris” vs “travelling to Paris”).
- Cannot generalize to questions outside the 3 predefined domains.
- High maintenance cost as patterns grow.

**Why it wasn’t enough:**  
It only works for narrow intents and fails for open-ended or paraphrased questions.

---

## 2. Full RAG System (LLM + Vector Store)
**Idea:** Use embeddings + a vector DB (FAISS, Pinecone, Chroma) + an LLM for final answer generation.

**Pros**
- Most flexible: can answer arbitrary questions.
- Produces natural, human-readable answers.
- Handles paraphrasing, synonyms, multi-hop reasoning.

**Cons**
- Requires GPU/LLM inference → not allowed in this assignment.
- More expensive infra.
- Risk of hallucinations unless carefully constrained.

**Why it wasn’t used:**  
The assignment explicitly restricts using LLMs for generation, and infrastructure complexity is unnecessary.

---

## 3. Hybrid “RAG-Lite” Retrieval + Rule-Based Answering (Chosen Approach)
**Idea:**  
- Use `fastembed` to embed all messages.  
- Retrieve top-K relevant snippets with cosine similarity.  
- Apply deterministic extraction logic on retrieved text.  
- If no intent matches, return retrieved snippets directly.

**Pros**
- Much more robust than pure regex.
- Handles paraphrasing (“headed to London” matches “trip to London next Friday”).
- Still deterministic and cheap to run.
- No LLM generation (compliant with the assignment).

**Cons**
- Retrieval quality depends on message density and user activity.
- Still limited to the 3 supported domains for structured answers.
- Cannot synthesize new statements.

**Why it was chosen:**  
This approach balances accuracy, generality, and system constraints while remaining simple to deploy.

---

## 4. (Rejected) Finite-State / Slot-Filling NLU Model
**Idea:** Train a small intent classifier + slot extractor locally.

**Pros**
- More flexible than regex.
- Small models could run CPU-only.

**Cons**
- Requires labeled training data (not provided).
- Harder to tune without real user variations.

**Why excluded:**  
Labeling and training overhead is out of scope for a short assignment.

---

## Final Choice Summary
I implemented a **hybrid system**:

- **Rule-based** extraction for high-precision answers  
- **Vector-based retrieval** to support paraphrased or generic questions  
- **No LLMs**, ensuring deterministic behavior and easy deployment (Render)

This gives the best mix of reliability and generalization within the project constraints.
