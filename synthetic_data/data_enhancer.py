import pandas as pd

def add_histroical_features(df):
    df.columns=df.columns.str.strip().str.lower() #ensures columns are clean

    #frequency features
    tg_counts=df['target_group_cleaned'].value_counts().to_dict()
    loc_counts=df['location_cleaned'].value_counts().to_dict()
    tg_loc_counts = df.groupby(['target_group_cleaned', 'location_cleaned']).size().to_dict()


    #avg loi per target group and location
    tg_avg_loi=df.groupby('target_group_cleaned')['loi_minutes'].mean().to_dict()
    loc_avg_loi=df.groupby('location_cleaned')['loi_minutes'].mean().to_dict()

    #Diversity: How many unique locations a TG has appeared in
    tg_diversity=df.groupby('target_group_cleaned')['location_cleaned'].nunique().to_dict()

    #Apply features to each row
    df['tg_freq']=df['target_group_cleaned'].map(tg_counts).fillna(0)
    df['loc_freq']=df['location_cleaned'].map(loc_counts).fillna(0)
    df['tg_loc_freq']=df.apply(lambda x: tg_loc_counts.get((x['target_group_cleaned'], x['location_cleaned']), 0), axis=1)
    df['tg_avg_loi']=df['target_group_cleaned'].map(tg_avg_loi).fillna(0)
    df['loc_avg_loi']=df['location_cleaned'].map(loc_avg_loi).fillna(0)
    df['tg_diversity']=df['target_group_cleaned'].map(tg_diversity).fillna(0)

    #Deviation features
    df['loi_deviation_tg']=df['loi_minutes']-df['tg_avg_loi']
    df['loi_deviation_loc']=df['loi_minutes']-df['loc_avg_loi']

    

    return df
if __name__=="__main__":
    df=pd.read_csv("output_final_cleaned.csv")
    enriched_df=add_histroical_features(df)
    enriched_df.to_csv("final_dataset_enriched.csv", index=False)
    print("historical features added and saved to final_dataset_enriched.csv")