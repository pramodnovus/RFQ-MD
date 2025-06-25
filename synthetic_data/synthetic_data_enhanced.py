#for synthetic data generation with enhanced features 
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import joblib


np.random.seed(42)
#loading enriched dataset
df=pd.read_csv("final_dataset_enriched.csv")
df.columns=df.columns.str.strip().str.lower()

df['is_feasible'] = 1

#embed text features
model=SentenceTransformer('all-MiniLM-L6-v2')
df['tg_vec']=df['target_group_cleaned'].apply(lambda x: model.encode(str(x)))
df['loc_vec']=df['location_cleaned'].apply(lambda x: model.encode(str(x)))

#create synthetic negatives (non-trivial)
num_synthetic=300
synthetic_rows=[]
targets=df['target_group_cleaned'].unique()
locations=df['location_cleaned'].unique()

for _ in range(num_synthetic):
    base=df.sample(1).iloc[0]
    tg=np.random.choice(targets)
    loc=np.random.choice(locations)

    #introduce more varied and realistic synthetic patterns
    loi_offset = np.random.normal(loc=15, scale=5)  # More dynamic offset
    loi = max(5, base['loi_minutes'] + loi_offset)

    # simulate unseen or rarely seen patterns intentionally
    tg_freq = np.random.choice([0, max(0, base['tg_freq'] - np.random.randint(3, 6))])
    loc_freq = np.random.choice([0, max(0, base['loc_freq'] - np.random.randint(3, 6))])
    tg_loc_freq = np.random.choice([0, base['tg_loc_freq']])
    tg_div = np.random.choice([0, max(0, base['tg_diversity'] - np.random.randint(1, 4))])

    synthetic_rows.append({
        'target_group_cleaned':tg,
        'location_cleaned':loc,
        'loi_minutes':round(loi,1),
        'tg_freq':tg_freq,
        'loc_freq':loc_freq,
        'tg_loc_freq':tg_loc_freq,
        'tg_diversity':tg_div,
        'loi_deviation_tg': abs(loi-base['tg_avg_loi']),
        'loi_deviation_loc': abs(loi-base['loc_avg_loi']),
        'is_feasible':0
    })


synthetic_df=pd.DataFrame(synthetic_rows)
combined_df=pd.concat([df, synthetic_df], ignore_index=True)

#Embed new rows
combined_df['tg_vec']=combined_df['target_group_cleaned'].apply(lambda x: model.encode(str(x)))
combined_df['loc_vec']=combined_df['location_cleaned'].apply(lambda x: model.encode(str(x)))

# ✅ Save a copy of the unscaled version for reasoning
combined_df.to_csv("synthetic+real_data_unscaled.csv", index=False)

#Normalize LOI+numeric features
scaler=MinMaxScaler()
numeric_cols=['loi_minutes','tg_freq', 'loc_freq', 'tg_loc_freq', 'tg_diversity', 'loi_deviation_tg', 'loi_deviation_loc']
combined_df[numeric_cols]=scaler.fit_transform(combined_df[numeric_cols])

#save scaler
joblib.dump(scaler,"enhanced_scaler.pkl")

#feature matrix
X=np.hstack([
    np.stack(combined_df['tg_vec']),
    np.stack(combined_df['loc_vec']),
    combined_df[numeric_cols].values
])
y=combined_df['is_feasible'].values
print("Label distribution in y:", combined_df['is_feasible'].value_counts(dropna=False))

np.save("X_enhanced.npy", X)
np.save("y_enhanced.npy", y)

combined_df.drop(columns=['tg_vec','loc_vec'],inplace=True)
combined_df.to_csv("synthetic+real_data.csv", index=False)

print("synthetic enhanced dataset created")