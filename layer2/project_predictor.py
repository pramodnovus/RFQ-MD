import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load model, scaler, and transformer once
model = joblib.load(r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\final_model\layer2\xgboost_model.joblib")
scaler = joblib.load(r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\final_model\synthetic_data\enhanced_scaler.pkl")
transformer = SentenceTransformer('all-MiniLM-L6-v2')

# Load historical data for reasoning
stats_df = pd.read_csv(r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\final_model\synthetic_data\synthetic+real_data_unscaled.csv")

def predict_feasibility(target_group, location, loi):
    """
    Predict feasibility, generate SHAP explanation, and return human-readable reasoning.
    
    Returns:
        prediction (0 or 1),
        confidence (float),
        explanation (str)
    """
    target_group = str(target_group).strip().lower()
    location = str(location).strip().lower()

    stats_df['target_group_cleaned'] = stats_df['target_group_cleaned'].astype(str).str.strip().str.lower()
    stats_df['location_cleaned'] = stats_df['location_cleaned'].astype(str).str.strip().str.lower()



    # Embed target group and location
    tg_vec = transformer.encode(str(target_group))
    loc_vec = transformer.encode(str(location))

    # Get historical stats
    tg_stats = stats_df[stats_df['target_group_cleaned'] == target_group]
    loc_stats = stats_df[stats_df['location_cleaned'] == location]
    tg_loc_stats = stats_df[
        (stats_df['target_group_cleaned'] == target_group) &
        (stats_df['location_cleaned'] == location)
    ]

    # Feature engineering
    tg_freq = tg_stats.shape[0]
    loc_freq = loc_stats.shape[0]
    tg_loc_freq = tg_loc_stats.shape[0]
    tg_div = tg_stats['location_cleaned'].nunique() if tg_freq > 0 else 0
    tg_avg_loi = tg_stats['loi_minutes'].mean() if tg_freq > 0 else loi
    loc_avg_loi = loc_stats['loi_minutes'].mean() if loc_freq > 0 else loi
    loi_dev_tg = abs(loi - tg_avg_loi)
    loi_dev_loc = abs(loi - loc_avg_loi)

    print("Target Group:", target_group)
    print("TG Avg LOI:", tg_avg_loi)
    print("TG Frequency:", tg_freq)

    numeric_input = np.array([[loi, tg_freq, loc_freq, tg_loc_freq, tg_div, loi_dev_tg, loi_dev_loc]])
    numeric_scaled = scaler.transform(numeric_input)

    # Final feature vector
    X = np.hstack([tg_vec, loc_vec, numeric_scaled.flatten()])

    # Model prediction
    prediction = model.predict([X])[0]
    confidence = model.predict_proba([X])[0][int(prediction)]

    # SHAP explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(np.array([X]))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("shap_explanation_from_pdf.png")

    # Human-readable explanation
    reasons = []

    if loi_dev_tg > 10:
        reasons.append(f"The LOI is {int(loi_dev_tg)} minutes different from the target group's average.")

    if tg_freq < 5:
        reasons.append(f"This target group appears only {tg_freq} times in past projects.")

    if loc_freq < 5:
        reasons.append(f"This location appears only {loc_freq} times in past projects.")

    if tg_loc_freq == 0:
        reasons.append("This target group and location combination has never been used together before.")

    if tg_div < 2:
        reasons.append(f"This target group has only been used in {tg_div} unique location(s).")

    if not reasons:
        reasons.append("No strong historical pattern was found for this project configuration.")

    explanation = " ".join(reasons)

    return prediction, confidence, explanation

