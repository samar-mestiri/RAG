### 1) Corrective Retrieval Augmented Generation (CRAG)
CRAG improves Retrieval-Augmented Generation by incorporating a self correction mechanism that evaluates and refines retrieved knowledge reducing errors and improving accuracy, while RAG retrieves documents and uses them to guide an LLM’s response. CRAG handles noisy, irrelevant or misleading data. It can be coupled with various RAG based approaches.

The technique is based upon a feedback loop that continuously evaluates the quality of retrieved documents and provides evaluation. 
### 2) Self RAG (Retrieval Augmented Generation)
Retrieval Augmented Generation combines a Large Language Model (LLM) with an external knowledge source to improve response accuracy and reduce hallucinations. Self Reflective Retrieval Augmented Generation builds on this by letting the model decide when to retrieve information using reflection tokens.

It evaluates its own outputs and only fetches external data when needed making the model more reliable and controllable during inference.