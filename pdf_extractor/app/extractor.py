import pdfplumber
import ollama
import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import re  # Added for JSON extraction

load_dotenv()  # Load the .env file

def extract_pdf_text(pdf_path):
    text = ""
    # Try pdfplumber text extraction first
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # If no text found, fallback to OCR
    if not text.strip():
        print(f"No text found via pdfplumber, running OCR on '{os.path.basename(pdf_path)}' ...")
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"

    return text

def extract_info_from_text(text):
    prompt = f"""
You are a strict JSON API that extracts information from client SOW documents.

From the text below, extract and return **only** the following fields in pure JSON format:
- target_group
- loi (length of interview)
- location (region/country/geography)
- Project Type

Return the output in this format (no explanation, no extra text):
{{
  "target_group": "...",
  "loi": "...",
  "location": "...",
  "project_type":"..."
}}

Text:
\"\"\"
{text}
\"\"\"
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # Extract only JSON from response, in case LLaMA adds explanation
    raw_output = response['message']['content']
    match = re.search(r'\{[\s\S]*?\}', raw_output)
    if match:
        return match.group(0)
    return raw_output.strip()
