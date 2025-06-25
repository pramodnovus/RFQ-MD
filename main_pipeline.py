import sys
import os
import pandas as pd

# Step 1: PDF Extraction
from pdf_extractor.app.extractor import extract_pdf_text, extract_info_from_text
from pdf_extractor.run_extract import process_extracted_json

# Step 2: Data Cleaning
from data_cleaning.output_cleaning import clean_fields

# Step 3: Similarity Matching
from layer1.layer1 import intelligent_match

# Step 4: ML Prediction (Layer 2)
from layer2.project_predictor import predict_feasibility

def main(pdf_path):
    print("\n--- Running Full Pipeline ---")

    # --- Step 1: PDF Extraction (disabled for now, using hardcoded test data) ---
    # raw_text = extract_pdf_text(pdf_path)
    # extracted_info_json = extract_info_from_text(raw_text)
    # extracted_fields = process_extracted_json(os.path.basename(pdf_path), extracted_info_json)
    # if not extracted_fields:
    #     print("Failed to extract fields from PDF.")
    #     return
    # print("\nExtracted Fields:")
    # print(extracted_fields)

    extracted_fields = {
        "target_group": "IT managers",
        "loi": "30 minutes",
        "location": "India, USA, Germany",
        "project_type": "CAVI"
}

    # --- Step 2: Clean extracted data ---
    print("\n cleaning extracted data...")
    df_input = pd.DataFrame([extracted_fields])
    cleaned_data = clean_fields(df_input)
    print(cleaned_data)

    # --- Step 3: Similarity Matching ---
    print("\n--- Similarity Match Result ---")
    result = intelligent_match(cleaned_data)

    if result['status'] == 'match_found':
        print(result['data'])
        print("Feasibility confirmed via similarity.")
        return

    # --- Step 4: ML Prediction ---
    print("No similar match found, using ML model...")
    tg = cleaned_data.loc[0, 'target_group_cleaned']
    loc = cleaned_data.loc[0, 'location_cleaned']
    loi = cleaned_data.loc[0, 'loi_minutes']

    pred, conf, explanation = predict_feasibility(tg, loc, loi)
    print(f"Target Group: {tg}")
    print(f"Prediction: {'Feasible' if pred else 'Infeasible'} (Confidence: {conf:.2f})")
    print("SHAP plot saved as 'shap_explanation_from_pdf.png'")
    print("Reasoning", explanation)

if __name__ == "__main__":
    test_pdf_path = r""  # Add your test file path if needed
    main(test_pdf_path)
