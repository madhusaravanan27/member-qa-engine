# Member QA – Design Notes

This project required building a question-answering system over a collection of member messages. Before choosing the final design, several different solution approaches were considered. This document explains those options and why the final architecture uses a hybrid approach.

## 1. Pure Rule-Based Extraction

The first idea was to answer everything with regex patterns: detecting trip-related questions, extracting locations and dates, counting cars, or identifying restaurant names.

Advantages:
- Very fast  
- Easy to debug  
- Fully deterministic  

Limitations:
- Breaks easily when the user asks the same question in a different way  
- Hard to scale to more question types  
- Not flexible enough for open-ended queries  

Because language varies a lot, this approach was not sufficient on its own.

## 2. Full RAG Using an LLM

A more powerful approach would be a full retrieval-augmented generation pipeline: embed messages, retrieve the most relevant ones, and let an LLM generate an answer.

Advantages:
- Handles natural language extremely well  
- Flexible enough for almost any question  

Limitations:
- Not allowed for this assignment  
- Requires more infrastructure  
- Adds unnecessary complexity  

This option was rejected due to project constraints.

## 3. Hybrid “RAG-Lite” (Final Choice)

The final solution combines two ideas:

- Rule-based logic for well-defined tasks (trip timing, car count, restaurants)  
- Embedding-based retrieval for open-ended questions  

This means the system gives exact answers when possible, and falls back to semantic search when the question does not match any rule.

Advantages:
- More flexible and robust than regex alone  
- Simple to run and deploy  
- No generative models required  
- Works even when the question is phrased differently  

This balance of precision and flexibility made it the best fit for the assignment.

## 4. Custom Intent Classifier

A small ML classifier could have been trained to detect question types.

Advantages:
- More generalizable than regex  
- Simple models would work  

Limitations:
- Requires labeled data  
- Still needs extraction logic afterwards  

This option was not selected because labeled training data was not available.

## 5. Summary of the Decision

After comparing all approaches, the hybrid solution was chosen because it provides reliable answers for structured queries and still handles broader, more natural questions through semantic retrieval. It meets all project constraints while remaining simple to deploy and maintain.
