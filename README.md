### 1) Corrective Retrieval Augmented Generation (CRAG)
CRAG improves Retrieval-Augmented Generation by incorporating a self correction mechanism that evaluates and refines retrieved knowledge reducing errors and improving accuracy, while RAG retrieves documents and uses them to guide an LLM’s response. CRAG handles noisy, irrelevant or misleading data. It can be coupled with various RAG based approaches.

The technique is based upon a feedback loop that continuously evaluates the quality of retrieved documents and provides evaluation. 
<br>
Why CRAG is Needed<br>
Below are the key reasons why CRAG is important in improving traditional RAG systems:

- Irrelevant Retrieval: Filters out documents that look similar but don’t answer the query.
- Noise and Errors: Detects and removes outdated or low quality information.
- Hallucinations: Validates retrieved context to reduce made-up or incorrect answers.
- Reliability: Ensures accurate and contextually correct information, critical for sensitive fields.
- Ranking of Documents: Re-ranks documents so the most relevant ones are prioritized.
- Dynamic Knowledge: Checks that retrieved data is current and relevant.
- Bias Reduction: Validates beyond similarity scores to prevent retrieval bias.
For example, when asking “What do koalas eat?”, RAG might retrieve documents about eucalyptus leaves but also mix in texts about pandas eating bamboo or kangaroos grazing grass, which can confuse the answer. CRAG adds a corrective step that filters out these irrelevant documents, keeping only accurate content about koalas so the final response clearly states they primarily eat eucalyptus leaves.

The step by step working of CRAG is mentioned below :

1. Input Query: The process begins with an input query like “What do koalas eat?”.

2. Retrieval (Vanilla RAG): The documents from knowledge base are selected based on their relevance to the input query. The retriever finds top K relevant documents based only on similarity.

3. Retrieval Evaluator: The relevance and quality of each document concerning the input query is assessed. The evaluator assigns a relevance score to each document.

4. Decision: Based on previous step, a decision is made.
Correct: If at least one document has a high relevance score then it is relevant and accurate.
Incorrect: If all documents have low relevance scores then they are irrelevant or incorrect.
Ambiguous: If the relevance scores are neither low nor high then there is uncertainty about the overall quality.

5. Corrective Step (if Correct): This ensures only the most accurate and context specific documents are kept.
Filter: Removes low quality or outdated documents.
Rerank: Combines similarity, quality and freshness to reorder docs.
Duplication: Prevents repeated or duplicated results.
6. Web Search (if Incorrect): If the documents are incorrect, web search is conducted to retrieve additional relevant information from the internet to make the knowledge base dynamic.

7. Combining Knowledge (if Ambiguous): If the documents are ambiguous, it combines both internal knowledge from initial retrieval and external knowledge from web search.

8. Answer Generation: The LLM uses uses only corrected, refined or newly retrieved information to generate more accurate and factual response.

The process starts by retrieving documents then using a corrective step to check their relevance and accuracy to the input query.

### 2) Self RAG (Retrieval Augmented Generation)
Retrieval Augmented Generation combines a Large Language Model (LLM) with an external knowledge source to improve response accuracy and reduce hallucinations. Self Reflective Retrieval Augmented Generation builds on this by letting the model decide when to retrieve information using reflection tokens.

It evaluates its own outputs and only fetches external data when needed making the model more reliable and controllable during inference.
#### How Self-RAG Outperforms Traditional RAG?
- Dynamic Retrieval: Adaptively decides when and how to retrieve information, rather than retrieving irrelevant and fixed amount of data.
- Self Critical Evaluation: Critiques its own generated responses to assess the quality and relevance of the retrieved documents and the generated response such that it is well supported by evidence from the retrieved document.
- Overcomes Hallucinations: Continuously checks and improves its answers to reduce chances of adding false or unsupported information.
- Lower Latency and Cost: Avoids unnecessary retrieval steps hence cutting cost and time required during running the pipeline. This ensures the system remains efficient while still maintaining access to external knowledge when truly necessary.
- High Scalability and Explainability: Reduces load by retrieving selectively, unlike RAG in which retrieval becomes slower and more expensive as the knowledge base grows.
While RAG offered improvements over plain LLMs, it remained inflexible, noisy, slow and costly. Self RAG addresses these issues by being adaptive, targeted, faster and more streamlined.