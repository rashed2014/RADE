# RADE: Retrieval-Augmented Document Entity Extractor

**RADE** is a document understanding framework that combines **retrieval-augmented generation (RAG)** with powerful entity extraction and question answering capabilities. It is optimized for **multi-page, multi-document** trust and legal documents, where key information such as grantors, trustees, beneficiaries, and legal metadata must be extracted reliably.

This implementation is inspired by the research paper:  
**["M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding"](https://arxiv.org/abs/2411.04952)**  
by *Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal*.

---

## 🧠 What is RADE?

RADE stands for **Retrieval-Augmented Document Entity Extractor**. It enables intelligent document processing by:
- Retrieving the most relevant sections of multi-page PDFs
- Parsing document content (from both text-based and scanned image PDFs)
- Extracting entities like **Grantor**, **Trustee**, **Beneficiary**, etc.
- Answering domain-specific questions with high accuracy

It integrates **ColBERT for retrieval**, **Azure Document Intelligence for parsing**, **GLiNER for entity extraction**, and **RoBERTa for question answering**, forming an end-to-end system for document analytics.

---

## ⚙️ Key Components

### 🔍 ColBERT (via RAGatouille)
ColBERT (Contextualized Late Interaction over BERT) is a dense retriever that allows fast and accurate top-k search over documents. It works by:
- Creating embeddings for queries and document passages separately
- Using **late interaction** to compute relevance scores efficiently
- Supporting fast retrieval via FAISS indexing

📦 Powered by the [RAGatouille](https://github.com/huggingface/RAGatouille) library for simplified setup and integration.

---

### 🧠 FAISS Indexing
FAISS (Facebook AI Similarity Search) is used to index dense vector embeddings of document pages:
- Supports exact and approximate nearest neighbor search
- Scalable to large document collections
- Integrates seamlessly with ColBERT in RAGatouille

---

### 📄 Azure Document Intelligence
Parsed documents—whether scanned images or text-based PDFs—are processed using **Azure's Document Intelligence API** to:
- Extract structured machine-readable text
- Merge paragraphs and tables
- Normalize content across scanned and native pages

---

### 🔎 GLiNER – Entity Extraction
[GLiNER](https://huggingface.co/knowledgator) is a general-purpose, encoder-based Named Entity Recognition model. It performs:
- Label-guided entity extraction
- Multi-span recognition
- Deduplicated results with confidence scoring

Used to extract structured entities like:
- `Grantor`
- `Trustee`
- `Beneficiary`
- `Successor Trustee`, etc.

---

### ❓ RoBERTa QA (HuggingFace)
The HuggingFace `pipeline("question-answering")` powered by **RoBERTa** is used to:
- Extract spans directly answering a natural language query
- Operate over retrieved and parsed context chunks
- Return ranked answers with confidence

---

## 🔁 RADE Workflow Overview

```text
     +------------------------+
     |   PDF (scanned/text)   |
     +-----------+------------+
                 |
                 v
       [Image/Page Parsing]
       Azure Document Intelligence
                 |
                 v
         Parsed page text
                 |
     +-----------+-------------+
     |                         |
     v                         v
ColBERT via RAGatouille   GLiNER Entity Extractor
     |                         |
     v                         |
Top-k relevant chunks          |
     +-----------+-------------+
                 |
            RoBERTa QA (Optional)
```

### 🚀 How to Use RADE
📁 1. Upload Your Files
Upload your .pdf document and a query_plan.json file that defines what you want to extract. Example:
```python
{
  "Who are the Grantors?": {
    "type": "ner",
    "labels": ["Grantor"]
  },
  "What is the name of the trust?": {
    "type": "qa",
    "labels": []
  }
}
```
### 🧱 2. Add and Index the Document

rade.add_document("path/to/document.pdf", doc_id="TrustDoc001")
rade.build_index()


### 🔎 3. Run Queries
```python
for query, meta in query_plan.items():
    results = rade.retrieve(query, k=5)

    if meta["type"] == "qa":
        qa_result = rade.run_qa_pipeline(query, results)
    else:
        ner_result = rade.extract_entities_with_gliner(results, meta["labels"], threshold=0.3)
```

### 📊 4. Visualize in Notebook

```python
show_query_results(result)
```
### 🧪 Citation
This work is built upon:

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal.
"M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding."
arXiv:2411.04952

