# Member QA – Design Notes and Data Insights

This document summarizes the full system design, the reasoning behind architectural choices, the insights gained from analyzing the member message dataset, and key details about the final deployed application. The complete system is live on Render and available for testing at:  
**https://member-qa-wwtb.onrender.com/docs#/**

## Design Notes

This project required building a question–answering system over a collection of member messages. Several approaches were explored before choosing the final hybrid solution.

### 1. Pure Rule-Based Extraction

One option was to rely entirely on regex-based extraction for identifying questions about trips, restaurants, favorites, or car counts. This method offered speed and predictability, but it failed as soon as user phrasing varied even slightly. Since it does not generalize well to conversational queries, it was too brittle to use alone.

### 2. Full Retrieval-Augmented Generation (RAG) With an LLM

A full RAG pipeline—embedding messages, retrieving documents, and using an LLM to write final answers—was also considered. This would have provided excellent natural language understanding and wide coverage. However, it was beyond the scope and constraints of this assignment, required heavier compute, and introduced unnecessary complexity. Thus, it was not selected.

### 3. Hybrid “RAG-Lite” Approach (Final Choice)

The final solution combines lightweight semantic retrieval with deterministic rule-based extraction. Structured question types (trip dates, yes/no travel questions, cars, restaurants, favorites) are handled with precise extraction rules, while ambiguous or loosely phrased queries benefit from retrieval-based matching. This hybrid approach offers strong coverage without the cost or overreach of a full LLM-based system.

### 4. Intent Classifier (Considered but Not Used)

A custom intent classifier could have replaced regex, offering more flexibility. However, training such a model requires labeled data, which was not available. Furthermore, even with classification, the system would still need rule-based extraction to produce answers. Due to these limitations, it was not implemented.

### 5. Final Decision Summary

The hybrid RAG-Lite system balances accuracy, robustness, and simplicity while staying within the assignment requirements. It handles structured questions with precision and open-ended questions with contextually relevant retrieval.

---

## Data Insights

### 1. Dataset Overview

A total of 600 messages were fetched from the upstream API. Data checks confirmed that the dataset was completely clean, with no missing fields, corrupted messages, or invalid timestamps. This level of quality supports reliable extraction and retrieval.

### 2. Message Length Patterns

All messages fell within normal conversational boundaries. There were no extremely short or excessively long entries, indicating that the dataset contains meaningful user content without noise or placeholder text.

### 3. Timestamp Validation

Timestamps ranged from November 2024 to November 2025. All were valid ISO-formatted datetime strings. This consistency would allow temporal reasoning if required.

### 4. User Activity Distribution

Message density varied widely across users. A small set of members accounted for a large share of the dataset:

- Vikram Desai – 70 messages  
- Sophia Al-Farsi – 66 messages  
- Armand Dupont – 62 messages  
- Lily O’Sullivan – 60 messages  
- Fatima El-Tahir – 59 messages  

This uneven distribution influences semantic retrieval, as active users naturally produce more context for embedding-based search.

### 5. Structural Consistency

There were no duplicate IDs, no malformed records, no missing text, and no invalid values. Encoding was consistent across the dataset. This suggests that the data was either manually curated or synthetically generated.

### 6. Final Observations

The dataset shows excellent overall consistency. The only noteworthy characteristic is uneven user activity distribution, which strengthens retrieval for some users and reduces it for others.

---

## API Endpoints

### `/ask`
This endpoint is designed for structured, intent-based questions, including:

- Trip-related questions  
- Trip summaries  
- Yes/No travel checks  
- Car ownership questions  
- Favorite restaurants  
- Favorite things  

The endpoint uses deterministic extraction rules first and then backs off to semantic retrieval only if needed.

### `/ask_generic`
This endpoint performs broad semantic retrieval without intent classification. It is ideal for exploratory questions, open-ended queries, or cases where the question does not clearly fall into a predefined category.

**Guidance**  
- Use `/ask` for trips, cars, restaurants, or favorites.  
- Use `/ask_generic` when the question is unclear or not part of the main intent domains.

---

## Outputs

### ask/ endpoint

<img width="878" height="278" alt="image" src="https://github.com/user-attachments/assets/ba847d57-a81c-4e70-8bdb-6f703a07fb97" />

<img width="873" height="217" alt="image" src="https://github.com/user-attachments/assets/fe7ab9dc-db5a-430c-b7c4-738ec7d5381e" />

<img width="883" height="229" alt="image" src="https://github.com/user-attachments/assets/13b9ee71-ffb3-4636-8136-ae58d4c34836" />

### ask/generic endpoint

<img width="857" height="237" alt="image" src="https://github.com/user-attachments/assets/27a052b7-f25b-4bf6-875f-f9296fd26e77" />

---

## Future Improvements

Several enhancements could extend the system’s capabilities. More advanced intent detection could reduce reliance on regex patterns, and lightweight machine-learning models could learn query types automatically. Extraction accuracy could be strengthened through NLP techniques such as dependency parsing or named entity recognition. Retrieval freshness could be improved through incremental indexing as new messages arrive. Summarization across multiple messages could produce richer answers when context is scattered. More descriptive error handling, retry logic for upstream services, and optional LLM-based answer refinement (if allowed) could make the system more robust and user-friendly.

---

## Conclusion

The Member QA system successfully delivers a flexible, accurate question-answering experience by combining rule-based extraction with a semantic retrieval layer. The hybrid RAG-Lite design ensures reliability for structured queries while enabling contextual search when questions are broader. The dataset’s cleanliness supports stable performance, and the endpoint design makes the system intuitive to integrate and test. The full system has been deployed on Render and can be accessed at:

**https://member-qa-wwtb.onrender.com/docs#/**

This deployment demonstrates that the solution is not only conceptually strong but also production-ready and easy to operate in a real environment.
