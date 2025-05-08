import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import cmd
import os
import traceback
from gliner import GLiNER
import os
import json
import pandas as pd
from ragatouille import RAGPretrainedModel
import fitz
from transformers import pipeline
import time
import re
from PIL import Image
import sys
from utils.azure_doc_intel import*
from dotenv import load_dotenv
from tqdm import tqdm

#load .env file
load_dotenv(dotenv_path=".env")

#wrapper classes
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


#define the RADE class
class RADE:
    def __init__(
        self,
        retrieval_model_name: str = "colbert-ir/colbertv2.0",
        qa_model_name: str = "deepset/roberta-base-squad2",  
        entity_extraction_model: str = "knowledgator/gliner-multitask-large-v0.5",
        max_pages: int = 4,
        use_approximate_index: bool = True,
        batch_size: int = 4,
        use_flash_attention: bool = False,  
        index_name: str = None,
        index_path: str = None,
    ):
        """Initialize M3DOCRAG framework with optimized multi-GPU support."""
        self.max_pages = max_pages
        self.use_approximate_index = use_approximate_index
        self.batch_size = batch_size
        self.index_name = index_name
        self.index_path = index_path
        # use your `key` and `endpoint` environment variables
        self.key = os.environ.get("key").strip()
        self.endpoint = os.environ.get("endpoint").strip()
        
        device_map = {
            "retrieval": "cuda:0",  
            "qa": "cuda:1"          
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
       
        # self.index = index_name
        self.pages: List[DocumentPage] = []
        print("Models initialized successfully!")

    def is_scanned_page(self, page: fitz.Page) -> bool: #need to handle scanned pages 
        return len(page.get_text("text").strip()) == 0


    def add_document(self, pdf_path: str, doc_id: str):
        print(f"📄 Loading document: {doc_id}")
        
        #set default index name to be the filename of the first pdf
        if self.index_name is None:
            self.index_name = pdf_path.split("/")[-1]

        #ensure file exsits
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Failed to open PDF '{pdf_path}': {e}")
            return  # Skip this document
        try:
            client = init_docintel_client(self.endpoint, self.key)
            json_str = parse_pdf(pdf_path, client)
            parsed_pages = json.loads(json_str)
        except Exception as e:
            print(f"⚠️ Azure parsing failed for doc is {doc_id}: {e}")
            retrun  # exit if parsing fails

        for page in parsed_pages:
            self.pages.append(DocumentPage(
                doc_id=doc_id,
                doc_name=pdf_path.split("/")[-1],
                page_num=page["page_number"],
                text=page["content_text"],
                is_scanned=False  # Since this method parses full PDF with layout, assume it's normalized
            ))


    def build_index(self):
        """Build the retrieval index for all pages."""
        print(f"🔧 Building index for {len(self.pages)} pages...")
    
        self.index_path = self.retrieval_model.index(
            collection=[page.text for page in self.pages],
            document_metadatas=[
                {"page": str(page.page_num), "document": page.doc_name} for page in self.pages
            ],
            index_name=self.index_name,
            max_document_length=180,
            split_documents=True,
            use_faiss=True,
            document_ids=[f"{page.doc_name}_page{page.page_num}" for page in self.pages]
        )
    
        print(f"✅ Index built: {self.index_path}")


    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPage]:
        """
        Retrieve ColBERT chunks and wrap them in DocumentPage objects.
    
        Returns:
            List[RetrievedPage]: Retrieved chunks with preserved text + metadata.
        """
        if not self.index_path:
            raise ValueError("Index not built. Call build_index() first.")
    
        print(f"🔍 Retrieving top-{k} chunks for query: {query}")
        results = self.retrieval_model.search(query=query, k=k)
    
        retrieved_pages = []
        for r in results:
            document_id = r.get("document_id", "")
            doc_name, page_num = None, None
    
            # Try to extract from expected format: 'docname.pdf_page5'
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
                is_scanned=None  # Unknown in this context
            )
    
            retrieved_pages.append(RetrievedPage(
                page=doc_page,
                score=r["score"]
            ))
    
        return retrieved_pages


    def run_qa_pipeline(self, question: str, retrieved_texts: List[Dict]) -> Dict:
        """
        Run the QA model on concatenated retrieved texts.
    
        Args:
            question (str): The question to answer.
            retrieved_texts (List[Dict]): List of retrieved chunks from the retriever, each with:
                - 'content': the retrieved passage
                - 'document_metadata': {'document': ..., 'page': ...}
    
        Returns:
            Dict: {
                'answer': str,
                'score': float,
                'retrieved': List[Dict] (original context chunks with metadata)
            }
        """
        if not retrieved_texts:
            return {
                "answer": "[No context retrieved]",
                "score": 0.0,
                "retrieved": []
            }
    
        # Clean and concatenate text
        combined_text = ". ".join(
            str(r.get("content", "")).replace("\n", " ") for r in retrieved_texts if r.get("content")
        )
    
        # Run QA
        qa_result = self.qa_pipeline(question=question, context=combined_text)
    
        # Attach retrieved contexts with source info
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
        Extract entities using GLiNER, filter by confidence score, and deduplicate.
    
        Args:
            retrieved_texts (List[Dict]): Retrieved passages with:
                - 'content': the chunk text
                - 'document_metadata': {'document': ..., 'page': ...}
            labels (List[str]): GLiNER entity labels to extract
            threshold (float): Minimum confidence score to keep an entity
    
        Returns:
            Dict: {
                "entities": [ { "text": ..., "label": ..., "score": ... }, ... ],
                "retrieved": [ { "content": ..., "document": ..., "page": ... }, ... ]
            }
        """
        if not retrieved_texts or not labels:
            return {"entities": [], "retrieved": []}
    
        combined_text = ". ".join(
            r.get("content", "").replace("\n", " ").replace("_", " ").strip()
            for r in retrieved_texts if r.get("content")
        )
    
        # Run GLiNER
        raw_entities = self.entity_extraction_model.predict_entities(combined_text, labels)
    
        # Filter by confidence score
        filtered = [
            {"text": e["text"], "label": e["label"], "score": e.get("score", 1.0)}
            for e in raw_entities
            if e.get("score", 1.0) >= threshold
        ]
    
        # Deduplicate by (text, label)
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

