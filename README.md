# Member QA – Design Notes and Data Insights

This document summarizes both the system design decisions and the findings from analyzing the member message dataset.

## Design Notes

This project required building a question-answering system over a collection of member messages. Several approaches were evaluated before choosing the final solution.

### 1. Pure Rule-Based Extraction

The simplest approach was to rely entirely on regex patterns to detect questions about trips, restaurants, or car counts.

**Advantages**
- Fast  
- Deterministic  
- Easy to debug  

**Limitations**
- Breaks when phrasing changes  
- Does not scale to new question types  
- Not suitable for open-ended or conversational queries  

This approach alone was too brittle.

### 2. Full Retrieval-Augmented Generation (RAG) With an LLM

Another option was to embed messages, retrieve relevant ones, and let an LLM generate the answer.

**Advantages**
- Handles natural language variations well  
- Very flexible  
- Works for almost any question type  

**Limitations**
- Not allowed for this assignment  
- Requires larger infrastructure  
- More complex than needed  

This was rejected due to project constraints.

### 3. Hybrid “RAG-Lite” Approach (Final Choice)

The final design combines two techniques:

- Rule-based logic for well-defined patterns (trip timing, car count, restaurants)  
- Embedding-based semantic retrieval for everything else  

This allows precise answers when the question is structured, and relevant context when the question is open-ended.

**Advantages**
- More reliable than pure regex  
- Lightweight and easy to deploy  
- Does not require generative models  
- Tolerant to varied question phrasing  

This approach offered the best balance of simplicity and robustness.

### 4. Intent Classifier (Considered but Not Used)

A model could classify questions into categories.

**Advantages**
- More flexible than regex  
- Learns patterns automatically  

**Limitations**
- Requires labeled training data  
- Still needs extraction logic for answers  

Due to lack of labeled data, this option was not selected.

### 5. Final Decision Summary

The hybrid RAG-Lite solution was chosen because it meets all project requirements, handles both structured and natural questions, and avoids unnecessary complexity.

---

## Data Insights

As part of the project, the dataset was analyzed to check for anomalies, inconsistencies, and general structure.

### 1. Dataset Overview

A total of 600 messages were fetched from the upstream API. Basic validation showed:

- No missing user names  
- No missing messages  
- No duplicate IDs  
- No invalid timestamps  
- No extremely short or extremely long messages  

The dataset is clean and well-structured.

### 2. Message Length Patterns

All messages fell into normal conversational ranges:

- 0 messages with 5 or fewer characters  
- 0 messages with 20 or fewer characters  
- 0 messages with 500 or more characters  

There was no spam, placeholder text, or corrupted entries.

### 3. Timestamp Validation

The timestamps ranged from November 2024 to November 2025. All timestamps were valid and well-formatted with no inconsistencies.

This means the dataset supports timeline-based features if needed.

### 4. User Activity Distribution

Message activity is uneven across users. The most active users were:

- Vikram Desai (70 messages)  
- Sophia Al-Farsi (66 messages)  
- Armand Dupont (62 messages)  
- Lily O’Sullivan (60 messages)  
- Fatima El-Tahir (59 messages)  

A small group of members contributes a large portion of the dataset.  
This affects retrieval density: more active users produce stronger semantic context.

### 5. Structural Consistency

There were no issues with:

- Invalid message IDs  
- Field formatting  
- Encoding  
- Missing fields  
- Duplicate timestamps  

The dataset appears either curated or synthetically generated due to its cleanliness.

### 6. Final Observations

The dataset is high-quality and shows no structural flaws.  
The only notable characteristic is the uneven distribution of messages across users, which influences retrieval performance.

## API Endpoints

### `/ask`
Handles **structured, intent-based** queries such as:
- Trip details  
- Trip summaries  
- Yes/No travel questions  
- Car counts  
- Favorite restaurants  
- Favorite things  

Uses rule-based extraction first, and falls back to semantic retrieval for these same domains.

### `/ask_generic`
A general semantic-search endpoint that retrieves top relevant messages for any open-ended question.

**When to use which**
- Use `/ask` for questions about trips, cars, restaurants, or favorites  
- Use `/ask_generic` for broad or exploratory questions  

## Outputs:
## ask/ endpoint:
<img width="878" height="278" alt="image" src="https://github.com/user-attachments/assets/ba847d57-a81c-4e70-8bdb-6f703a07fb97" />

<img width="873" height="217" alt="image" src="https://github.com/user-attachments/assets/fe7ab9dc-db5a-430c-b7c4-738ec7d5381e" />

<img width="883" height="229" alt="image" src="https://github.com/user-attachments/assets/13b9ee71-ffb3-4636-8136-ae58d4c34836" />

## ask/generic endpoint:

<img width="857" height="237" alt="image" src="https://github.com/user-attachments/assets/27a052b7-f25b-4bf6-875f-f9296fd26e77" />





