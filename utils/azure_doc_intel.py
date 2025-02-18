#azure imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from io import BytesIO
import base64  

# use your `key` and `endpoint` environment variables
import os
key = os.environ.get("AZURE_DOC").strip()
endpoint = os.environ.get("AZURE_DOC_AI_ENDPOINT").strip()
 
def parse_azureDocIntell(image):  
    all_lines = []
    all_figures = []
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_encoded_pdf = base64.b64encode(buffer.getvalue()).decode("utf-8")

    analyze_request = {
        "base64Source": base64_encoded_pdf
    }
    
    document_intelligence_client = DocumentIntelligenceClient(
         endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout", analyze_request=analyze_request
    )
    
    result = poller.result()
    
    for page in result.pages:

    
        if page.lines:
            for line in page.lines:
                all_lines.append(line.content)

            
    all_pages = ". ".join(all_lines)
    return all_pages
        

