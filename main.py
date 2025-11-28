import os
import re
import io
import json
import time
import fitz
import requests
import tempfile
import uvicorn
from typing import List
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY missing from .env")

genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-pro"

app = FastAPI(title="HackRx Bill Extraction API")

def download_document(path_or_url: str) -> str:
    """Downloads remote URL or copies local path → temp file"""
    if os.path.exists(path_or_url):
        _, ext = os.path.splitext(path_or_url)
        if not ext: ext = ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            with open(path_or_url, "rb") as f:
                tmp.write(f.read())
            return tmp.name

    try:
        r = requests.get(path_or_url, timeout=25)
        r.raise_for_status()

        content_type = r.headers.get("content-type", "")
        if "pdf" in content_type:
            ext = ".pdf"
        elif "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type:
            ext = ".jpg"
        else:
            ext = ".pdf" # Default fallback

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(r.content)
            return tmp.name
    except Exception as e:
         raise HTTPException(400, f"Failed to download: {str(e)}")


def pdf_to_images(file_path: str) -> List[Image.Image]:
    """Convert PDF → list of PIL images"""
    doc = fitz.open(file_path)
    pages = []
    for p in doc:
        pix = p.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append(img)
    doc.close()
    return pages

EXTRACTION_PROMPT = """
Extract line items from this bill page.

RULES:
- Extract ONLY line items (ignore totals)
- For each item, extract:
  - item_name (exact)
  - qty
  - rate
  - amount (net)
- Detect page type: "Pharmacy", "Bill Detail", "Final Bill"

OUTPUT FORMAT (NOT JSON):

PAGE <page_no> - <PAGE_TYPE>
1. <item_name> | qty:<qty> | rate:<rate> | amount:<amount>
2. <item_name> | qty:<qty> | rate:<rate> | amount:<amount>
"""

generation_config = genai.types.GenerationConfig(
    temperature=0,
    top_p=1,
    top_k=1,
)

def ask_gemini(img: Image.Image, page_no: int):
    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=generation_config
    )

    response = model.generate_content(
        contents=[
            EXTRACTION_PROMPT.replace("<page_no>", str(page_no)),
            img
        ]
    )
    
    return response.text, response.usage_metadata

def parse_page_type(text: str) -> str:
    m = re.search(r"PAGE\s*\d+\s*-\s*(Pharmacy|Bill Detail|Final Bill)", text, re.IGNORECASE)
    if not m:
        return "Bill Detail"
    label = m.group(1).strip()
    if "pharm" in label.lower():
        return "Pharmacy"
    if "final" in label.lower():
        return "Final Bill"
    return "Bill Detail"


ITEM_REGEX = re.compile(
    r"\d+\.\s*(.*?)\s*\|\s*qty\s*:\s*([\d\.]+)\s*\|\s*rate\s*:\s*([\d\.]+)\s*\|\s*amount\s*:\s*([\d\.]+)",
    re.IGNORECASE
)


def parse_items(text: str):
    items = []
    for m in ITEM_REGEX.finditer(text):
        name = m.group(1).strip()

        try:
            qty = float(m.group(2))
        except: qty = 1.0
            
        try:
            rate = float(m.group(3))
        except: rate = 0.0
            
        try:
            amt = float(m.group(4))
        except: amt = 0.0

        items.append({
            "item_name": name,
            "item_amount": amt,
            "item_rate": rate,
            "item_quantity": qty
        })
    return items

@app.post("/extract-bill-data")
async def extract_bill_data(req: dict):
    document_url = req.get("document")
    if not document_url:
        raise HTTPException(400, "Missing 'document' field")

    temp_path = None

    try:
        temp_path = download_document(document_url)

        pages = pdf_to_images(temp_path)

        pagewise_output = []
        total_items = 0

        total_input_tokens = 0
        total_output_tokens = 0

        for i, page_img in enumerate(pages, start=1):
            text, usage = ask_gemini(page_img, i)
            
            if usage:
                total_input_tokens += usage.prompt_token_count
                total_output_tokens += usage.candidates_token_count

            page_type = parse_page_type(text)
            items = parse_items(text)
            total_items += len(items)

            pagewise_output.append({
                "page_no": str(i),
                "page_type": page_type,
                "bill_items": items
            })

        return {
            "is_success": True,
            "token_usage": {
                "total_tokens": total_input_tokens + total_output_tokens,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            },
            "data": {
                "pagewise_line_items": pagewise_output,
                "total_item_count": total_items
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "is_success": False,
            "data": None,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "error_msg": str(e)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
