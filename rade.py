import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline
)

from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from PIL import Image
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import cmd
import os
import traceback
from gliner import GLiNER

#wrapper classes
@dataclass
class DocumentPage:
    """Represents a single page from a document with its metadata."""
    doc_id: str
    page_num: int
    image: Image.Image  

@dataclass
class RetrievedPage:
    """Represents a retrieved page with its relevance score."""
    page: DocumentPage
    score: float

#define the RADE class
class RADE:
    def __init__(
        self,
        retrieval_model_name: str = "vidore/colpali-v1.3",
        qa_model_name: str = "deepset/roberta-base-squad2",  
        entity_extraction_model: str = "knowledgator/gliner-multitask-large-v0.5",
        max_pages: int = 4,
        use_approximate_index: bool = True,
        batch_size: int = 4,
        use_flash_attention: bool = False,  
    ):
        """Initialize M3DOCRAG framework with optimized multi-GPU support."""
        self.max_pages = max_pages
        self.use_approximate_index = use_approximate_index
        self.batch_size = batch_size
        
        if torch.cuda.device_count() > 1:
            device_map = {
                "retrieval": "cuda:0",  
                "qa": "cuda:1"          
            }
        else:
            device_map = {
                "retrieval": "cuda",  
                "qa": "cuda"          
            }
        print(f"Using device map: {device_map}")

        print("Initializing retrieval model...")
        self.retrieval_model = ColPali.from_pretrained(
            retrieval_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device_map["retrieval"]}
        ).eval()
            
        self.retrieval_processor = ColPaliProcessor.from_pretrained(retrieval_model_name)
        self.retrieval_device = device_map["retrieval"]
        self.qa_device = device_map["qa"]

        print("Initializing QA model and entity extraction models...")

        self.qa_model = pipeline("question-answering", model=qa_model_name, device=self.qa_device) 
        self.entity_extraction_model = GLiNER.from_pretrained(entity_extraction_model, device=self.qa_device)
       
        self.index = None
        self.pages: List[DocumentPage] = []
        print("Models initialized successfully!")

    def add_document(self, pdf_path: str, doc_id: str):
        """Add a PDF document to the corpus."""
        print(f"Loading document: {doc_id}")
        page_images = convert_from_path(pdf_path, dpi=144)
        for page_num, image in enumerate(page_images):
            page = DocumentPage(
                doc_id=doc_id,
                page_num=page_num,
                image=image
            )
            self.pages.append(page)
        print(f"Added {len(page_images)} pages from {doc_id}")

    def build_index(self):
        """Build the retrieval index for all pages."""
        print(f"Building index for {len(self.pages)} pages...")
        
        total_batches = (len(self.pages) + self.batch_size - 1) // self.batch_size
        
        dataloader = DataLoader(
            self.pages,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.retrieval_processor.process_images([page.image for page in x])
        )

        all_embeddings = []
        try:
            for batch_doc in tqdm(dataloader, desc="Building index", total=total_batches):
                with torch.no_grad():
                    batch_doc = {
                        k: v.to(self.retrieval_device, dtype=torch.bfloat16) 
                        if k == "pixel_values" else v.to(self.retrieval_device)
                        for k, v in batch_doc.items()
                    }
                    
                    with torch.cuda.device(self.retrieval_device):
                        torch.cuda.empty_cache()
                    
                    embeddings_doc = self.retrieval_model(**batch_doc)
                    embeddings_doc = embeddings_doc.to(dtype=torch.float32)
                    embeddings_doc = embeddings_doc.mean(dim=1)
                    
                    embeddings_doc = embeddings_doc.cpu()
                    all_embeddings.extend(list(torch.unbind(embeddings_doc)))

            all_embeddings = np.stack([emb.numpy() for emb in all_embeddings])
            print(f"Embeddings shape: {all_embeddings.shape}")
            embedding_dim = all_embeddings.shape[1]
            n_vectors = all_embeddings.shape[0]

            if self.use_approximate_index and n_vectors >= 156:
                print("Building approximate index...")
                quantizer = faiss.IndexFlatIP(embedding_dim)
                n_centroids = max(1, min(
                    n_vectors // 40,
                    int(np.sqrt(n_vectors)),
                    100
                ))
                print(f"Using {n_centroids} centroids for IVF index")
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    embedding_dim,
                    n_centroids,
                    faiss.METRIC_INNER_PRODUCT
                )
                self.index.train(all_embeddings)
            else:
                print("Building exact index...")
                self.index = faiss.IndexFlatIP(embedding_dim)

            self.index.add(all_embeddings)
            print("Index built successfully!")
            
        except Exception as e:
            print(f"Error during index building: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            raise

    def retrieve(self, queries: str) -> List[RetrievedPage]:
        """Retrieve relevant pages."""

        retrieved_pages = [[] for _ in queries]
        try:
            ### **Step 1: Retrieve Pages for All Queries**
            query_batch = self.retrieval_processor.process_queries(queries).to(self.retrieval_device)

            with torch.no_grad():
                query_batch = {k: v.to(self.retrieval_device) for k, v in query_batch.items()}
                query_embedding= self.retrieval_model(**query_batch).to(dtype=torch.float32)
                query_embedding_np = query_embedding.cpu().numpy().mean(axis=1)
            
            # Run FAISS search for queries
            scores, indices = self.index.search(query_embedding_np, self.max_pages)
            
            # Store retrieved pages for queries
            for query_idx, (score_list, index_list) in enumerate(zip(scores, indices)):
                for score, idx in zip(score_list, index_list):
                    if idx < len(self.pages):
                        retrieved_pages[query_idx].append(RetrievedPage(page=self.pages[idx], score=float(score)))
            return retrieved_pages

        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return []

    def run_qa_pipeline(self, question: str, retrieved_texts: str) -> List[Dict]:
        """
        Run a QA model on retrieved texts.
        
        Args:
            question (str): The question to answer (e.g., "What are the names of the grantors?").
            retrieved_texts (List[Dict]): A list of retrieved text in JSON format, e.g.,
                [
                    {"text": "The grantor is John Doe.", "score": 0.95},
                    {"text": "Grantor: Alice Johnson", "score": 0.85},
                    {"text": "The trustee is Bob Smith.", "score": 0.80},
                ]
    
        Returns:
            List[Dict]: List of answers with scores and contexts.
        """
        qa_result = self.qa_model(question=question, context=retrieved_texts)
    
        return qa_result

    def extract_entities_with_gliner(self, retrieved_texts: str, labels: List[str]) -> List[Dict]:
        """
        Extract entities using GLiNER and remove duplicates.
    
        Args:
            retrieved_texts (List[Dict]): Retrieved texts.
            labels (List[str]): Target labels.
    
        Returns:
            List[Dict]: Deduplicated extracted entities.
        """
        entities = self.entity_extraction_model.predict_entities(retrieved_texts, labels)
        
        # Deduplicate entities by converting to a set of tuples
        unique_entities = { (entity["text"], entity["label"]) for entity in entities }
    
        # Convert back to a list of dictionaries
        deduplicated_entities = [{"text": text, "label": label} for text, label in unique_entities]
        return deduplicated_entities
