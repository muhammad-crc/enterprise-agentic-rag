# 📊 Enterprise Agentic RAG — Quality Evaluation Report

**Generated At:** `2026-10-19 01:40:45`  

**LLM Evaluator:** `openai/gpt-oss-120b` (via Groq)  

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  

**Total Benchmark Questions:** `9`  

**Total Run Time:** `174.2s`  


## 1. Metric Overview

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Faithfulness** | **`0.6667`** | Measures how factually grounded the answer is in the retrieved context (no hallucination). |
| **Answer Relevancy** | **`0.6587`** | Measures how directly and completely the answer addresses the user question. |
| **Context Precision** | **`1.0`** | Measures whether relevant chunks are ranked high at the top of the context list. |
| **Context Recall** | **`1.0`** | Measures whether the retriever found all necessary context to match the ground truth. |

---

## 2. Detailed Test Cases

### Q1: What is the purpose of this dummy PDF file?

- **Expected (Ground Truth):** This dummy PDF file was created for testing RAG pipelines.
- **Agent Answer:** The dummy PDF file was created specifically to test Retrieval‑Augmented Generation (RAG) pipelines, serving as a test document for evaluating how well the vector database retrieves chunks and how accurately the LLM generates answers without hallucination. [1]  ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q2: What does Section 1 of the document test?

- **Expected (Ground Truth):** Section 1 tests retrieval, specifically how well the vector database finds this chunk.
- **Agent Answer:** Section 1 tests the retrieval capability, specifically how well the vector database can locate that particular chunk of text.    ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q3: How does the document test the vector database?

- **Expected (Ground Truth):** Section 1 tests how well the vector database can find and retrieve the text chunk.
- **Agent Answer:** The document includes a **Retrieval** section (Section 1) that explicitly tests the vector database by checking how well it can locate the specific chunk of text.    ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q4: What does Section 2 evaluate in the RAG pipeline?

- **Expected (Ground Truth):** Section 2 tests generation, specifically if the LLM answers accurately without hallucination.
- **Agent Answer:** Section 2 evaluates the generation step of the RAG pipeline, specifically testing whether the LLM provides accurate answers without hallucination. [1]  ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q5: How does the document verify that the LLM does not hallucinate?

- **Expected (Ground Truth):** Section 2 tests generation to check if the LLM answers accurately without hallucination.
- **Agent Answer:** I could not find this information in the provided documents.
- **Retrieved Chunks:** `0` chunk(s)


### Q6: What are the two main sections described in the test document?

- **Expected (Ground Truth):** Section 1 on retrieval testing vector database search and Section 2 on generation testing accurate LLM responses without hallucination.
- **Agent Answer:** The test document describes two main sections:  - **Section 1: Retrieval** – tests how well the vector database finds the chunk.   - **Section 2: Generation** – tests if the LLM answers accurately without hallucination.    ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q7: Does Section 1 test generation or retrieval?

- **Expected (Ground Truth):** Section 1 tests retrieval.
- **Agent Answer:** Section 1 tests retrieval, not generation. It is described as “Retrieval tests how well the vector database finds this chunk.”    ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q8: Does Section 2 test vector database chunk finding or generation accuracy?

- **Expected (Ground Truth):** Section 2 tests generation accuracy to ensure the LLM answers without hallucination.
- **Agent Answer:** Section 2 tests generation accuracy, not vector‑database chunk finding.    ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


### Q9: What pipeline architecture is being tested by this sample PDF?

- **Expected (Ground Truth):** RAG (Retrieval-Augmented Generation) pipelines.
- **Agent Answer:** The sample PDF is used to test a **Retrieval‑Augmented Generation (RAG) pipeline**, checking both the vector‑database retrieval step and the downstream LLM generation for accuracy and hallucination‑free responses.    ### Sources & Citations - **dummy_rag_test.pdf**, Page 1 [1]
- **Retrieved Chunks:** `1` chunk(s)
- **Citations:** `dummy_rag_test.pdf` (p. 1)


---

## 3. How to Re-Run Evaluation

To execute this evaluation benchmark after modifying the retriever, chunking, or agent workflow:

```powershell
cd "enterprise-agentic-rag"
python -m evaluation.evaluate_rag
```
