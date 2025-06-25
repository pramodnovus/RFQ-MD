from sentence_transformers import SentenceTransformer, util
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pycountry
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapi")
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_connection():
    return psycopg2.connect(
        dbname="my_project",
        user="postgres",
        password="ittil@123",
        host="localhost",
        port="5432"
    )


def intelligent_match(new_project):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM extracted_projects", conn)
    conn.close()

    # Drop incomplete rows
    df = df.dropna(subset=["target_group_cleaned", "location_cleaned", "loi_minutes"]).reset_index(drop=True)

    # Convert new project fields to string
    new_text = new_project["target_group_cleaned"] + " " + new_project["location_cleaned"]
    new_embedding = model.encode(new_text, convert_to_tensor=True)

    db_texts = (df["target_group_cleaned"] + " " + df["location_cleaned"]).tolist()
    db_embeddings = model.encode(db_texts, convert_to_tensor=True)

    # Text similarity
    sim_scores = util.cos_sim(new_embedding, db_embeddings)[0].cpu().numpy()

    # LOI similarity
    loi_array = df["loi_minutes"].astype(float).to_numpy()
    new_loi = float(new_project["loi_minutes"])
    loi_diff = np.abs(loi_array - new_loi)
    loi_sim = 1 - (loi_diff / df["loi_minutes"].max())

    # Match length
    min_len = min(len(sim_scores), len(loi_sim))
    sim_scores = sim_scores[:min_len]
    loi_sim = loi_sim[:min_len]

    final_score = 0.7 * sim_scores + 0.3 * loi_sim

    df = df.iloc[:min_len].copy()
    df["similarity_score"] = final_score

    best_idx = df["similarity_score"].idxmax()
    best_match = df.loc[best_idx]
    best_score = best_match["similarity_score"]
    loi_diff_from_best = abs(best_match["loi_minutes"] - new_loi)
    
    if best_score > 0.75 and loi_diff_from_best <= 5:
        return {
        "status": "match_found",
        "data": best_match
    }
    else:
        return {
        "status": "no_match_found",
        "data": None
    }

# Test case
if __name__ == "__main__":
    new_project = {
        "target_group_cleaned": "health manufacturers",
        "location_cleaned": "California",
        "loi_minutes": 30
    }

    result = intelligent_match(new_project)

    if result["status"] == "match_found":
        print("\nSimilar project found:\n")
        print(result["data"])
    else:
        print("\nNo similar project found. Proceed to ML model.\n")
