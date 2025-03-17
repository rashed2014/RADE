# RADE: Retrieval-Augmented Document Entity Extractor

This implementation is inspired by the research ["M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding"](https://arxiv.org/abs/2411.04952) by Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal.

---

## Overview

RADE (Retrieval-Augmented Document Entity Extractor) is a multi-page, multi-document processing framework designed to efficiently process and extract information from large-scale document collections. It combines ColPali-based multi-modal retrieval with encoder-based models for entity extraction and question answering.

---

## Key Components

- **ColPali Retriever:**  
  Retrieves relevant document pages using multi-modal embeddings, allowing text-based queries to retrieve image-based documents.

- **FAISS Indexing:**  
  Indexes image-based page embeddings. Supports both exact and approximate modes for scalability.

- **Document Parsing:**  
  Retrieved document pages (in image format) are parsed using **Azure Document Intelligence** to extract machine-readable text.

- **GLiNER (Encoder Model):**  
  Extracts key entities (e.g., Grantors, Trustees, Beneficiaries) from parsed text.

- **HuggingFace RoBERTa QA Pipeline:**  
  Performs extractive question-answering tasks over parsed document text.

---

## Workflow Overview

1. PDF documents are converted into image pages.
2. ColPali processes page images and creates embeddings, indexed via FAISS.
3. User queries are embedded and relevant pages retrieved.
4. Retrieved pages are parsed using Azure Document Intelligence.
5. Parsed text is passed to GLiNER for entity extraction and RoBERTa QA for answering queries.

---

## Example Usage

Refer to the provided `RADE_Pipeline.ipynb` notebook for the full workflow demonstration including document retrieval, parsing, entity extraction, and question answering.

---

## Benefits

- Combines multi-modal retrieval with encoder-based models.
- Handles multi-page, multi-document scenarios.
- Preserves visual information during retrieval.
- Uses scalable FAISS indexing.
- Modular and extensible.

---

## Citation

We build upon the methods proposed in M3DOCRAG by Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal.


