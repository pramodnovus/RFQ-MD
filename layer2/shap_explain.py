import numpy as np
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

#step 1: load model and data
X=np.load("X.npy")
model=joblib.load("xgboost_model.joblib")

#use treeexplainer(xgboost is tree based)
explainer=shap.Explainer(model)
shap_values=explainer(X)

#step 3: summary plot( top features across all samples)
print("Generating shap summary plot")
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap summary plot.png")
print("saved as shap_summary_plot.png")

#optional: force plot for single example
print("generating shap force plot for a random sample")
shap.initjs()
sample_idx=0 #change this index to test different rows
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.savefig("shap_waterfall_plot.png")
print("saved waterfall plot")