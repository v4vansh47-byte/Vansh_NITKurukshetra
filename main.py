import os
import json
import time
import tempfile
import shutil
import requests
import uvicorn
import easyocr
import numpy as np
import fitz
from typing import List, Literal, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, BeforeValidator
from typing_extensions import Annotated
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-2.5-pro"

reader = easyocr.Reader(['en'], gpu=False)

app = FastAPI(title="HackRx Bill Extractor")

def parse_float(v: Any) -> float:
    if v is None: return 0.0
    if isinstance(v, (float, int)): return float(v)
    if isinstance(v, str):
        clean_v = v.replace(",", "").replace("$", "").replace("₹", "").replace("Rs.", "").strip()
        if not clean_v: return 0.0
        try: return float(clean_v)
        except ValueError: return 0.0
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

def perform_ocr(file_path: str) -> List[str]:
    ocr_results = []
    doc = fitz.open(file_path)
    
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=300)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img_np = np.ascontiguousarray(img_np[..., :3])
            
        text_list = reader.readtext(img_np, detail=0)
        full_text = "\n".join(text_list)
        ocr_results.append(full_text)
        
    doc.close()
    return ocr_results

def analyze_text_with_retry(ocr_text: str, page_num: int, retries=3):
    system_instruction = """
    You are a data extractor. I will give you OCR text from a medical bill.
    
    TASK: Find the itemized table in the text and extract rows.
    
    RULES:
    1. **Format**: JSON matching the schema.
    2. **Noise**: OCR text might be messy. Ignore headers/footers.
    3. **Columns**:
       - 'item_name': Description.
       - 'item_amount': Net Amount (Rate * Qty).
    4. **Exclusion**: Do NOT extract "Total" or "Subtotal" lines as items.
    """

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": PageLineItems,
        },
        system_instruction=system_instruction
    )

    prompt = f"Extract items from Page {page_num}. OCR TEXT:\n\n{ocr_text}"

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            print(f"API Error (Attempt {attempt+1}): {e}")
            time.sleep(2)

    raise ValueError("Gemini failed after retries.")

@app.post("/extract-bill-data", response_model=APIResponse, response_model_exclude_none=True)
async def extract_bill_data(request: UserRequest):
    temp_file_path = None
    try:
        temp_file_path = download_file(request.document)
        
        print("Starting OCR extraction...")
        page_texts = perform_ocr(temp_file_path)
        print(f"OCR Complete. Found {len(page_texts)} pages.")

        extracted_pages = []
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0

        for i, text_content in enumerate(page_texts):
            if not text_content.strip(): continue
            
            gemini_response = analyze_text_with_retry(text_content, i+1)
            
            page_data = json.loads(gemini_response.text)
            
            page_data['page_no'] = str(i+1)
            extracted_pages.append(page_data)
            
            if gemini_response.usage_metadata:
                usage = gemini_response.usage_metadata
                total_tokens += usage.total_token_count
                input_tokens += usage.prompt_token_count
                output_tokens += usage.candidates_token_count

        total_count = sum(len(p.get('bill_items', [])) for p in extracted_pages)
        
        token_usage = TokenUsage(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return APIResponse(
            is_success=True,
            token_usage=token_usage,
            data=ExtractionData(
                pagewise_line_items=extracted_pages
            ),
            total_item_count=total_count
        )

    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        return APIResponse(is_success=False, data=None, error_msg=str(e))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
