import os
import json
import time
import tempfile
import shutil
import requests
import uvicorn
from typing import List, Literal, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, BeforeValidator
from typing_extensions import Annotated
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-2.5-pro"

app = FastAPI(title="HackRx Bill Extractor")

def parse_float(v: Any) -> float:
    if v is None: return 0.0
    if isinstance(v, (float, int)): return float(v)
    if isinstance(v, str):
        clean_v = v.replace(",", "").replace("$", "").replace("₹", "").replace("Rs.", "").strip()
        if not clean_v: return 0.0
        try: return float(clean_v)
        except: return 0.0
    return 0.0

FlexibleFloat = Annotated[float, BeforeValidator(parse_float)]

class BillItem(BaseModel):
    item_name: str = Field(..., description="Name of the item")
    item_amount: FlexibleFloat = Field(..., description="Net Amount")
    item_rate: FlexibleFloat = Field(0.0, description="Unit Rate")
    item_quantity: FlexibleFloat = Field(1.0, description="Quantity")

class PageLineItems(BaseModel):
    page_no: str = Field(..., description="Page number as STRING")
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy", "Summary", "Unknown"]
    bill_items: List[BillItem] = Field(default_factory=list)

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]

class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

class APIResponse(BaseModel):
    is_success: bool
    token_usage: Optional[TokenUsage] = None
    data: Optional[ExtractionData] = None
    total_item_count: Optional[int] = None
    error_msg: Optional[str] = None

class UserRequest(BaseModel):
    document: str

def download_file(url_or_path: str) -> str:
    r"""Handles local paths and URLs."""
    if os.path.exists(url_or_path):
        _, ext = os.path.splitext(url_or_path)
        if not ext: ext = ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copy2(url_or_path, tmp.name)
            return tmp.name

    try:
        response = requests.get(url_or_path, stream=True, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        ext = '.pdf' if 'pdf' in content_type else '.jpg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def analyze_document_with_retry(file_path: str, retries=3):
    mime_type = "application/pdf" if file_path.endswith(".pdf") else "image/jpeg"
    
    uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
    
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(1)
        uploaded_file = genai.get_file(uploaded_file.name)
    
    if uploaded_file.state.name == "FAILED":
        raise ValueError("Gemini failed to process file.")

    system_instruction = """
    You are a data extractor. 
    TASK: Extract every single row from the item tables in this document.
    RULES:
    1. Extract all items, medicines, or services found in tables.
    2. Do NOT ignore valid items. 
    3. Columns: Name, Rate, Qty, Amount.
    4. Exclude: 'Grand Total' or 'Sub Total' lines if possible.
    """

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": 0.1, 
            "response_mime_type": "application/json",
            "response_schema": list[PageLineItems], 
        },
        system_instruction=system_instruction
    )

    for attempt in range(retries):
        try:
            response = model.generate_content(
                [uploaded_file, "Extract all table rows."],
                request_options={"timeout": 600}
            )
            try: genai.delete_file(uploaded_file.name)
            except: pass
            return response
        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(2)

    try: genai.delete_file(uploaded_file.name)
    except: pass
    raise ValueError("Gemini API failed.")

@app.post("/extract-bill-data", response_model=APIResponse, response_model_exclude_none=True)
async def extract_bill_data(request: UserRequest):
    temp_file_path = None
    try:
        temp_file_path = download_file(request.document)

        gemini_response = analyze_document_with_retry(temp_file_path)
        extracted_pages = json.loads(gemini_response.text)
        
        total_count = sum(len(p.get('bill_items', [])) for p in extracted_pages)
        
        usage = gemini_response.usage_metadata
        token_usage = TokenUsage(
            total_tokens=usage.total_token_count,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count
        )

        return APIResponse(
            is_success=True,
            token_usage=token_usage,
            data=ExtractionData(pagewise_line_items=extracted_pages),
            total_item_count=total_count
        )

    except Exception as e:
        print(f"Error: {e}")
        return APIResponse(is_success=False, data=None, error_msg=str(e))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
