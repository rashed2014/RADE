import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import os
import json
import pandas as pd
from ragatouille import RAGPretrainedModel
from transformers import pipeline
from gliner import GLiNER
import fitz
from PIL import Image
from dotenv import load_dotenv
from utils.azure_doc_intel import init_docintel_client, parse_pdf

# Optional: load environment variables from a .env file
# load_dotenv(dotenv_path=".env")

# Wrapper classes
@dataclass
class DocumentPage:
    doc_id: str
    doc_name: str
    page_num: int
    text: str
    is_scanned: bool

@dataclass
class RetrievedPage:
    page: DocumentPage
    score: float
    rank: int

class RADE:
    def __init__(
        self,
        retrieval_model_name: str = "colbert-ir/colbertv2.0",
        qa_model_name: str = "deepset/roberta-base-squad2",
        entity_extraction_model: str = "knowledgator/gliner-multitask-large-v0.5",
        doc_int_key: str = "",
        doc_int_endpoint: str = "",
        max_pages: int = 3,
        batch_size: int = 32,
        use_flash_attention: bool = False,
        index_name: str = None,
        index_path: str = None,
        use_faiss: bool = False
    ):
        """Initialize RADE framework with retrieval, QA, and entity extraction models."""
        self.max_pages = max_pages
        self.batch_size = batch_size
        self.index_name = index_name
        self.index_path = index_path
        self.use_faiss = use_faiss

        # Load Azure Document Intelligence credentials
        if not doc_int_key or not doc_int_key.strip():
            raise ValueError("❌ RADE Initialization failed: 'key' must not be empty.")
        if not doc_int_endpoint or not doc_int_endpoint.strip():
            raise ValueError("❌ RADE Initialization failed: 'endpoint' must not be empty.")
        
        self.doc_int_key = doc_int_key
        self.doc_int_endpoint = doc_int_endpoint
        

        device_map = {
            "retrieval": "cuda",
            "qa": "cuda"
        }

        print(f"Using device map: {device_map}")
        print("Initializing retrieval model...")
        self.retrieval_model = RAGPretrainedModel.from_pretrained(retrieval_model_name)
        self.retrieval_device = device_map["retrieval"]
        self.qa_device = device_map["qa"]

        print("Initializing QA model and entity extraction models...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            tokenizer=qa_model_name,
            device=self.qa_device
        )

        self.entity_extraction_model = GLiNER.from_pretrained(entity_extraction_model, device=self.qa_device)
        self.pages: List[DocumentPage] = []
        print("Models initialized successfully!")

    def add_document(self, pdf_path: str, doc_id: str):
        """
        Parse and ingest a document into the system.
        Uses Azure Document Intelligence to extract page-level text.
        """
        print(f"📄 Loading document: {doc_id}")
        if self.index_name is None:
            self.index_name = pdf_path.split("/")[-1]

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Failed to open PDF '{pdf_path}': {e}")
            return

        try:
            client = init_docintel_client(self.doc_int_endpoint, self.doc_int_key)
            json_str = parse_pdf(pdf_path, client)
            parsed_pages = json.loads(json_str)
        except Exception as e:
            print(f"⚠️ Azure parsing failed for doc {doc_id}: {e}")
            return

        for page in parsed_pages:
            self.pages.append(DocumentPage(
                doc_id=doc_id,
                doc_name=pdf_path.split("/")[-1],
                page_num=page["page_number"],
                text=page["content_text"],
                is_scanned=False
            ))

    def build_faiss_index(self):
        """Index parsed pages using FAISS backend (if enabled)."""
        print(f"🔧 Building index for {len(self.pages)} pages...")
        self.index_path = self.retrieval_model.index(
            collection=[page.text for page in self.pages],
            document_metadatas=[
                {"page_num": str(page.page_num), "document_name": page.doc_name} for page in self.pages
            ],
            index_name=self.index_name,
            max_document_length=180,
            split_documents=True,
            use_faiss=self.use_faiss,
            document_ids=[f"{page.doc_name}_page{page.page_num}" for page in self.pages]
        )
        print(f"✅ Index built: {self.index_path}")

    def encode_docuemnt(self):
        """
        Encode documents in memory using ColBERT retriever.
        Used as an alternative to FAISS indexing.
        """
        print(f"Encoding document in memory...")
        if self.use_faiss:
            print("⚠️ use_faiss is True. Use build_faiss_index instead.")
            return

        all_texts = []
        all_metadatas = []
        for page in self.pages:
            if not page.text:
                continue
            all_texts.append(page.text)
            all_metadatas.append({
                "document": page.doc_name,
                "page": str(page.page_num)  # ✅ Keep 1-based page_num
            })

        self.retrieval_model.encode(all_texts, document_metadatas=all_metadatas)


    def retrieve_faiss_index(self, queries: List[str], k: int = 3) -> List[RetrievedPage]:
        """
        Retrieve top-k results using FAISS index for each query.
        """
        if not self.index_path:
            raise ValueError("Index not built. Call build_faiss_index() first.")

        retrieved_pages = []
        for query in queries:
            results = self.retrieval_model.search(query=query, k=k)
            for r in results:
                document_id = r.get("document_id", "")
                doc_name, page_num = None, None
                try:
                    doc_name, page_str = document_id.rsplit("_page", 1)
                    page_num = int(page_str)
                except Exception:
                    doc_name = document_id

                doc_page = DocumentPage(
                    doc_id=document_id,
                    doc_name=doc_name,
                    page_num=page_num,
                    text=r["content"],
                    is_scanned=None
                )
                retrieved_pages.append(RetrievedPage(
                    page=doc_page,
                    score=r["score"],
                    rank=r["rank"]
                ))

        return retrieved_pages


    def search_encoded_docs(self, queries: List[str], k: int = 3, batch_size: int = 32) -> List[RetrievedPage]:
        """
        Retrieve top-k results using in-memory encoded documents.
        """
        retrieved_pages = []
        for query in queries:
            results = self.retrieval_model.search_encoded_docs(query=query, k=k, bsize=batch_size)

            for r in results:
                metadata = r.get("document_metadata", {})
                doc_name = metadata.get("document", "unknown")
                page_str = metadata.get("page", "1")  # ✅ default to page 1 if missing

                try:
                    page_num = int(page_str)
                except ValueError:
                    page_num = 1

                doc_page = DocumentPage(
                    doc_id=f"{doc_name}_page{page_num}",
                    doc_name=doc_name,
                    page_num=page_num,  # ✅ 1-based
                    text=r.get("content", ""),
                    is_scanned=None
                )

                retrieved_pages.append(RetrievedPage(
                    page=doc_page,
                    score=r.get("score", 0.0),
                    rank=r.get("rank", 0)
                ))

        return retrieved_pages




    def run_qa_pipeline(self, question: str, retrieved_texts: List[Dict]) -> Dict:
        """
        Run QA model on concatenated retrieved contexts.
        """
        if not retrieved_texts:
            return {"answer": "[No context retrieved]", "score": 0.0, "retrieved": []}

        combined_text = ". ".join(
            str(r.get("content", "")).replace("\n", " ") for r in retrieved_texts if r.get("content")
        )

        qa_result = self.qa_pipeline(question=question, context=combined_text)
        qa_result["retrieved"] = [
            {
                "content": r.get("content", ""),
                "document": r.get("document_metadata", {}).get("document", "N/A"),
                "page": r.get("document_metadata", {}).get("page", "N/A")
            }
            for r in retrieved_texts
        ]
        return qa_result

    def extract_entities_with_gliner(self, retrieved_texts: List[Dict], labels: List[str], threshold: float = 0.3) -> Dict:
        """
        Extract entities from retrieved passages using GLiNER.
        - Filters by confidence threshold and removes duplicates.
        """
        if not retrieved_texts or not labels:
            return {"entities": [], "retrieved": []}

        combined_text = ". ".join(
            r.get("content", "").replace("\n", " ").replace("_", " ").strip()
            for r in retrieved_texts if r.get("content")
        )

        raw_entities = self.entity_extraction_model.predict_entities(combined_text, labels)

        filtered = [
            {"text": e["text"], "label": e["label"], "score": e.get("score", 1.0)}
            for e in raw_entities if e.get("score", 1.0) >= threshold
        ]

        seen = set()
        deduplicated = []
        for ent in filtered:
            key = (ent["text"], ent["label"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(ent)

        return {
            "entities": deduplicated,
            "retrieved": [
                {
                    "content": r.get("content", ""),
                    "document": r.get("document_metadata", {}).get("document", "N/A"),
                    "page": r.get("document_metadata", {}).get("page", "N/A")
                }
                for r in retrieved_texts
            ]
        }
