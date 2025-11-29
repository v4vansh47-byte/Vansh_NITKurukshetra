# HackRx Bill Extraction API

This project is a high-performance, AI-powered API designed to extract line-item details from complex medical bills and pharmacy invoices.

It leverages **Google Gemini 2.5 Pro (Multimodel)** to accurately identify table rows, extract financial data, and structure it into JSON, while strictly adhering to accounting rules (preventing double-counting of totals).

## 🚀 Live Deployment
**Base URL:** `https://vansh-nitkurukshetra.onrender.com`
**Endpoint:** `POST /extract-bill-data`

## ✨ Key Features

* **Multimodal Extraction:** Direct processing of PDFs and Images (JPG, PNG) without intermediate OCR text conversion, preserving spatial layout context.
* **Intelligent Validation:**
    * Extracts individual line items (`Medicine`, `Consultation`, `Room Charges`).
    * **Prevents Double Counting:** Explicitly identifies and excludes "Sub-total", "Brought Forward", and "Grand Total" lines from the item list.
* **Memory Efficient:** Optimized to run on lightweight cloud instances (removed heavy dependencies like PyTorch/EasyOCR to fit within 512MB RAM limits).
* **Strict Schema Compliance:** The output format matches the HackRx Postman collection exactly.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Framework:** FastAPI
* **AI Engine:** Google Gemini 2.5 Pro (via `google-generativeai`)
* **Validation:** Pydantic
* **Deployment:** Render (Cloud Hosting)

## 📝 API Documentation

### Extract Bill Data
Extracts structured line items from a document URL.

* **URL:** `/extract-bill-data`
* **Method:** `POST`
* **Content-Type:** `application/json`

#### Request Body
```json
{
  "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_2.png?sv=2025-07-05&spr=https&st=2025-11-24T14%3A13%3A22Z&se=2026-11-25T14%3A13%3A00Z&sr=b&sp=r&sig=WFJYfNw0PJdZOpOYlsoAW0XujYGG1x2HSbcDREiFXSU%3D"
}
```
#### Request Body
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
