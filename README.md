# HackRx Bill Extraction API

This project is a high-performance, AI-powered API designed to extract line-item details from complex medical bills and pharmacy invoices.

The system now uses **OCR (EasyOCR + PyMuPDF)** to extract raw text from PDFs and images before sending the cleaned text to **Google Gemini 2.5 Pro** for structured line-item extraction.

This hybrid pipeline significantly improves accuracy on noisy scans, multi-page PDFs, and low-quality medical bills.

---

## 🚀 Live Deployment

**Hosted On:** Hugging Face Spaces  
**Base URL:** `https://vansh-nitkurukshetra.hf.space`  
**Endpoint:** `POST /extract-bill-data`

---

## ✨ Key Features

### 🔍 **OCR + AI Hybrid Extraction**
- Uses **PyMuPDF** to convert PDF pages into high-resolution images.
- Runs **EasyOCR** on each page to extract accurate page-wise text.
- Sends OCR output to **Gemini 2.5 Pro** for:
  - Table row interpretation  
  - Item name extraction  
  - Amount, rate, and quantity detection  
  - Page-type classification

### 🧠 **Intelligent Financial Validation**
- Extracts detailed line items such as:  
  *Medicine, Diagnostics, Consultation, Room Charges, Consumables*
- **Prevents double counting** by excluding:
  - *Sub-total*
  - *Brought Forward*
  - *Grand Total*
- Handles broken tables, multi-line items, and missing columns.

### ⚡ **Optimized for Hugging Face Spaces**
- No GPU required
- Works smoothly within Space CPU-only environment
- Memory-friendly OCR pipeline suitable for free HF Space plans

### 📦 **Schema-Strict Output**
Matches the **HackRx Postman collection format exactly**, including:
- page type
- validated line items
- item counts
- token usage summary

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI
- **OCR Engine:** EasyOCR
- **PDF Engine:** PyMuPDF (fitz)
- **AI Model:** Google Gemini 2.5 Pro (via `google-generativeai`)
- **Validation:** Pydantic (v2)
- **Hosting:** Hugging Face Spaces (FastAPI Space)

---

## 📝 API Documentation

### 📌 Extract Bill Data  
Extracts structured line items from a document URL or local file path.

- **URL:** `/extract-bill-data`
- **Method:** `POST`
- **Content-Type:** `application/json`

---

### ✅ Request Body

```json
{
  "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_2.png?sv=2025-07-05&spr=https&st=2025-11-24T14%3A13%3A22Z&se=2026-11-25T14%3A13%3A00Z&sr=b&sp=r&sig=WFJYfNw0PJdZOpOYlsoAW0XujYGG1x2HSbcDREiFXSU%3D"
}
### ✅ Response Body

```json
{
    "is_success": "boolean",
    "token_usage": {
        "total_tokens": "integer"
        "input_tokens": "integer",
        "output_tokens": "integer"
    },
    "data": {
        "pagewise_line_items": [
            {
                "page_no": "string",
                "page_type": "Bill Detail | Final Bill | Pharmacy",
                "bill_items": [
                    {
                        "item_name": "string",
                        "item_amount": "float",
                        "item_rate": "float",
                        "item_quantity": "float"
                    }
                ]
            }
        ],
        "total_item_count": "integer"
    }
}
