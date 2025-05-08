import os
import json
from typing import Tuple, Dict, List
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult

def init_docintel_client(endpoint: str, key: str) -> DocumentIntelligenceClient:
    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
        connection_timeout=600
    )

def parse_paragraphs(analyze_result: AnalyzeResult) -> Tuple[Dict[int, List[dict]], List[int]]:
    table_offsets = []
    page_content = {}

    for paragraph in analyze_result.paragraphs:  
        for span in paragraph.spans:
            if span.offset not in table_offsets:
                for region in paragraph.bounding_regions:
                    page_number = region.page_number
                    if page_number not in page_content:
                        page_content[page_number] = []
                    page_content[page_number].append({
                        "content_text": paragraph.content
                    })
    return page_content, table_offsets

def parse_tables(analyze_result: AnalyzeResult, table_offsets: List[int]) -> Dict[int, List[dict]]:
    page_content = {}

    for table in analyze_result.tables:
        table_data = []
        for region in table.bounding_regions:
            page_number = region.page_number
            for cell in table.cells:
                for span in cell.spans:
                    table_offsets.append(span.offset)
                table_data.append(f"Cell [{cell.row_index}, {cell.column_index}]: {cell.content}")

        if page_number not in page_content:
            page_content[page_number] = []
        
        page_content[page_number].append({
            "content_text": "\n".join(table_data)
        })
    
    return page_content

def combine_paragraphs_tables(filepath: str, paragraph_content: Dict[int, List[dict]], table_content: Dict[int, List[dict]]) -> List[dict]:
    page_content_concatenated = {}
    structured_data = []

    for p_number in set(paragraph_content.keys()).union(table_content.keys()):
        concatenated_text = ""

        if p_number in paragraph_content:
            for content in paragraph_content[p_number]:
                concatenated_text += content["content_text"] + "\n"

        if p_number in table_content:
            for content in table_content[p_number]:
                concatenated_text += content["content_text"] + "\n"
        
        page_content_concatenated[p_number] = concatenated_text.strip()

    for p_number, concatenated_text in page_content_concatenated.items():
        structured_data.append({
            "page_number": p_number,
            "content_text": concatenated_text,
            "pdf_file": os.path.basename(filepath)
        })

    return structured_data

def parse_pdf(filepath: str, client: DocumentIntelligenceClient) -> List[dict]:
    """
    Full Azure Document Intelligence PDF parsing pipeline.
    Returns structured data with text grouped by page.
    """
    with open(filepath, "rb") as file:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file.read())
        )
        analyze_result: AnalyzeResult = poller.result()

    paragraph_content, table_offsets = parse_paragraphs(analyze_result)
    table_content = parse_tables(analyze_result, table_offsets)
    structured_data = combine_paragraphs_tables(filepath, paragraph_content, table_content)
    # Convert the structured data to JSON format
    return json.dumps(structured_data, indent=4)

