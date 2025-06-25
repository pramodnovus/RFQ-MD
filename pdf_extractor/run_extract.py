import os
import psycopg2
import json
import re
import sys
import time
from datetime import datetime
from .app.extractor import extract_pdf_text, extract_info_from_text

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def save_debug_info(pdf_file, stage, data):
    """Save debug information for troubleshooting"""
    debug_dir = "debug_info"
    os.makedirs(debug_dir, exist_ok=True)
    filename = f"{pdf_file}_{stage}.txt"
    with open(os.path.join(debug_dir, filename), "w", encoding="utf-8") as f:
        f.write(str(data))

def clean_json_string(json_str):
    """Fix common JSON formatting issues with more aggressive cleaning"""
    try:
        # Remove JSONP wrapper if exists
        json_str = re.sub(r'^\s*\w+\s*\(|\)\s*;\s*$', '', json_str)
        
        # Fix common issues
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Trailing commas
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', json_str)  # Unquoted keys
        json_str = json_str.replace("'", '"')  # Single quotes
        json_str = re.sub(r':\s*([a-zA-Z_]\w*)(\s*[,}])', r': "\1"\2', json_str)  # Unquoted string values
        json_str = re.sub(r'\\[^u]', r'\\\\', json_str)  # Fix backslashes
        
        # Try to parse to validate
        json.loads(json_str)
        return json_str
    except:
        return json_str

def extract_fields(data):
    """Flexible field extraction with support for nested values"""
    def deep_search(key, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == key.lower():
                    return v
                if isinstance(v, (dict, list)):
                    result = deep_search(key, v)
                    if result is not None:
                        return result
        elif isinstance(obj, list):
            for item in obj:
                result = deep_search(key, item)
                if result is not None:
                    return result
        return None

    def flatten(value):
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Flatten dictionary values recursively
            flat_values = []
            for val in value.values():
                if isinstance(val, list):
                    flat_values.extend(val)
                else:
                    flat_values.append(val)
            return ", ".join(str(v) for v in flat_values)
        elif value is None:
            return ""
        return str(value).strip()

    target_group = flatten(
        deep_search("target_group", data) or
        deep_search("target", data) or
        deep_search("targetGroup", data)
    )

    loi = flatten(
        deep_search("loi", data) or
        deep_search("length_of_interview", data) or
        deep_search("interview_length", data) or
        deep_search("duration", data)
    )

    location = flatten(
        deep_search("location", data) or
        deep_search("region", data) or
        deep_search("country", data) or
        deep_search("geography", data)
    )

    project_type= flatten(
        deep_search("project_type", data) or
        deep_search("project type", data)
    )

    return {
        "target_group": target_group,
        "loi": loi,
        "location": location,
        "project_type": project_type
    }

def process_extracted_json(pdf_file, extracted_info_json):
    """Enhanced JSON processing with more detailed error handling"""
    # Save raw output for debugging
    save_debug_info(pdf_file, "raw", extracted_info_json)
    
    # First attempt - direct parse
    try:
        data = json.loads(extracted_info_json)
        save_debug_info(pdf_file, "direct_parse", data)
        fields = extract_fields(data)
        if fields and any(fields.values()):  # Changed to any() to be more lenient
            return fields
    except json.JSONDecodeError as e:
        log(f"Direct parse failed: {str(e)}")
    
    # Second attempt - clean then parse
    try:
        cleaned_json = clean_json_string(extracted_info_json)
        data = json.loads(cleaned_json)
        save_debug_info(pdf_file, "cleaned_parse", data)
        fields = extract_fields(data)
        if fields and any(fields.values()):
            return fields
    except json.JSONDecodeError as e:
        log(f"Cleaned parse failed: {str(e)}")
    
    # Third attempt - extract JSON-like object
    try:
        matches = re.findall(r'\{[^{}]*\}', extracted_info_json)
        for match in matches:
            try:
                data = json.loads(match)
                fields = extract_fields(data)
                if fields and any(fields.values()):
                    save_debug_info(pdf_file, "regex_extracted", data)
                    return fields
            except:
                continue
    except Exception as e:
        log(f"Regex extraction failed: {str(e)}")
    
    # Final fallback - try to parse as plain text
    try:
        text_data = {
            "text": extracted_info_json
        }
        fields = extract_fields(text_data)
        if fields and any(fields.values()):
            return fields
    except Exception as e:
        log(f"Text fallback failed: {str(e)}")
    
    return None

def main():
    log("=== STARTING PROCESSING ===")
    sys.stdout = open(r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\pdf_extractor\processing.log", "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    
    try:
        conn = psycopg2.connect(
            dbname="my_project",
            user="postgres",
            password="ittil@123",
            host="localhost"
        )
        conn.autocommit = True
        cur = conn.cursor()

        pdf_folder = r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\pdf_extractor\pdf\2024 SoW"
        
        # Get all PDF files and sort them
        existing_files = sorted([
            f for f in os.listdir(pdf_folder)
            if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(pdf_folder, f))
        ])

        log(f"Found {len(existing_files)} files to process")
        
        # Skip first 122 files (0-121) and start from 122 (which is index 122, the 123rd file)
        files_to_process = existing_files[279:]
        
        log(f"Resuming from file 123 (skipping first 121 files)")
        log(f"Files remaining to process: {len(files_to_process)}")
        
        for i, pdf_file in enumerate(files_to_process, 280):  # Start counting from 123
            file_success = False
            log(f"\n--- Processing {i}/{len(existing_files)}: {pdf_file} ---")
            
            try:
                # Step 1: Extract text
                pdf_path = os.path.join(pdf_folder, pdf_file)
                pdf_text = extract_pdf_text(pdf_path)
                
                if not pdf_text.strip():
                    log("Warning: Empty text extracted")
                    continue
                
                # Step 2: Extract info using LLM
                extracted_info_json = extract_info_from_text(pdf_text)
                
                # Step 3: Process JSON with enhanced handling
                fields = process_extracted_json(pdf_file, extracted_info_json)
                if not fields:
                    log("Warning: Could not extract valid fields from JSON")
                    log(f"JSON content was:\n{extracted_info_json[:1000]}...")  # Log first 1000 chars
                    continue
                
                # Step 4: Insert into DB (now with more lenient requirements)
                name = os.path.splitext(pdf_file)[0]
                cur.execute("""
                    INSERT INTO extracted_projects (name, target_group, loi, location, project_type)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    name,
                    fields.get("target_group", ""),
                    fields.get("loi", ""),
                    fields.get("location", ""),
                    fields.get("project_type", "")
                ))
                
                log(f"Successfully processed: {name}")
                file_success = True
                
            except Exception as e:
                log(f"Error processing file: {str(e)}")
                import traceback
                log(traceback.format_exc())
                continue
            
            finally:
                if not file_success:
                    with open("failed_files.txt", "a") as f:
                        f.write(f"{pdf_file}\n")
    
    except Exception as e:
        log(f"Fatal error: {str(e)}")
        import traceback
        log(traceback.format_exc())
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()
        log("\n=== PROCESSING COMPLETED ===")

if __name__ == "__main__":
    main()