import os
import re
import time
import tempfile
import shutil
import requests
import uvicorn
from typing import List, Literal, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-2.5-pro" 

app = FastAPI(title="HackRx Bill Extractor")

# --- 2. SCHEMAS ---
class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageLineItems(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: Optional[TokenUsage] = None
    data: Optional[ExtractionData] = None
    error_msg: Optional[str] = None

class UserRequest(BaseModel):
    document: str

# --- 3. UTILS ---
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

# --- 4. REGEX PARSING LOGIC ---
def parse_gemini_text_response(text: str):
    """
    Parses the raw text response using Regex.
    Expected format from AI: "1 | Paracetamol | 50.0 | 2.0 | 100.0"
    """
    items = []
    # Regex Pattern: Page | Name | Rate | Qty | Amount
    pattern = re.compile(r"^\d+\s*\|\s*(.*?)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)", re.MULTILINE)

    for match in pattern.finditer(text):
        try:
            name = match.group(1).strip()
            rate = float(match.group(2))
            qty = float(match.group(3))
            amount = float(match.group(4))
            
            items.append(BillItem(
                item_name=name,
                item_amount=amount,
                item_rate=rate,
                item_quantity=qty
            ))
        except:
            continue
            
    return items

# --- 5. AI ENGINE (Regex Mode) ---
def analyze_document_regex(file_path: str, retries=3):
    mime_type = "application/pdf" if file_path.endswith(".pdf") else "image/jpeg"
    uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
    
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(1)
        uploaded_file = genai.get_file(uploaded_file.name)
    
    if uploaded_file.state.name == "FAILED":
        raise ValueError("Gemini failed.")

    system_instruction = """
    You are a bill extractor. Extract table rows.
    
    OUTPUT FORMAT (Strict Text):
    For every item row found, output a single line in this format:
    PageNumber | ItemName | Rate | Quantity | NetAmount
    
    Example Output:
    1 | Paracetamol | 10.50 | 2.0 | 21.00
    1 | Consultation | 500.0 | 1.0 | 500.00
    
    RULES:
    1. Do not use JSON. Use the pipe separator format above.
    2. Extract all items.
    3. Ignore Grand Totals.
    """

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"temperature": 0.1}, 
        system_instruction=system_instruction
    )

    for attempt in range(retries):
        try:
            response = model.generate_content([uploaded_file, "Extract items."])
            try: genai.delete_file(uploaded_file.name)
            except: pass
            return response
        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(2)

    try: genai.delete_file(uploaded_file.name)
    except: pass
    raise ValueError("Gemini failed.")

# --- 6. ENDPOINT ---
@app.post("/extract-bill-data", response_model=APIResponse, response_model_exclude_none=True)
async def extract_bill_data(request: UserRequest):
    temp_file_path = None
    try:
        temp_file_path = download_file(request.document)
        
        # 1. Get Raw Text from AI
        gemini_response = analyze_document_regex(temp_file_path)
        raw_text = gemini_response.text
        
        # 2. Parse Text with Regex
        items = parse_gemini_text_response(raw_text)
        
        # 3. Construct Response 
        page_data = PageLineItems(
            page_no="1",
            page_type="Bill Detail",
            bill_items=items
        )
        
        usage = gemini_response.usage_metadata
        token_usage = TokenUsage(
            total_tokens=usage.total_token_count,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count
        )

        return APIResponse(
            is_success=True,
            token_usage=token_usage,
            data=ExtractionData(
                pagewise_line_items=[page_data],
                total_item_count=len(items)
            )
        )

    except Exception as e:
        print(f"Error: {e}")
        return APIResponse(is_success=False, data=None, error_msg=str(e))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
