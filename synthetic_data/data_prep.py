#preparing data by normalizing and all before synthetic data has been generated
import pandas as pd
df=pd.read_csv("output_final_cleaned.csv")
df['is_feasible']=1
df=df.dropna(subset=["target_group_cleaned", "location_cleaned", "loi_minutes"])
from sentence_transformers import SentenceTransformer
model=SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np 
df['tg_vec']=df['target_group_cleaned'].apply(lambda x: model.encode(str(x)))
df['loc_vec']=df['location_cleaned'].apply(lambda x: model.encode(str(x)))

tg_array=np.stack(df['tg_vec'].values)
loc_array=np.stack(df['loc_vec'].values)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
loi_scaled=scaler.fit_transform(df[['loi_minutes']])

X=np.hstack([tg_array, loc_array, loi_scaled])
y=df['is_feasible'].values

np.save("X.npy", X)
np.save("y.npy", y)

import joblib
joblib.dump(scaler, "loi_scaler.pkl")

df.to_pickle("df_with_vectors.pkl")
df=pd.read_pickle("df_with_vectors.pkl")
print(df['tg_vec'].iloc[0])

df.drop(columns=['tg_vec', 'loc_vec'], inplace=True)
df.to_csv("output_final_cleaned.csv", index=False)