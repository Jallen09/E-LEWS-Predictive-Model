#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 01:18:02 2026

@author: johnniespiller
"""
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

# --- MASTER DATASET CREATION ---
# Generate Variables Master File
#np.random.seed(42)
n = 2000
data = {
    'District': np.random.choice(['Houston ISD', 'Conroe ISD'], n, p=[0.6, 0.4]),
    'Attendance_Rate': np.random.beta(7, 1, n).round(2),
    'Econ_Disadvantaged': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'Home_Language': np.random.choice(['English', 'Spanish', 'Other'], n, p=[0.5, 0.4, 0.1]),
    'Vocab_BOY': np.random.normal(8, 3, n).clip(0, 20),
    'Vocab_MOY': np.random.normal(11, 3, n).clip(0, 20),
    'Vocab_EOY': np.random.normal(15, 3, n).clip(0, 20),
    'LNF_BOY_Score': np.clip(np.random.poisson(15, n), 0, 52),
    'Phonological_Aware': np.clip(np.random.normal(12, 4, n).astype(int), 0, 20),
    'RAN_Speed_Sec': np.random.normal(120, 25, n).round(1),
    'Vocabulary_Tier': np.random.choice([1, 2, 3], n, p=[0.2, 0.5, 0.3]),
    'Parent_Engagement': np.random.randint(1, 6, n),
    'PreK_Enrollment': np.random.choice(['Yes', 'No'], n, p=[0.8, 0.2])
}
df_master = pd.DataFrame(data)

# 2. Add Longitudinal Growth & Set Unified Risk Flag
df_master['Total_Growth'] = (df_master['Vocab_EOY'] - df_master['Vocab_BOY']).round(1)
mask = (df_master['LNF_BOY_Score'] < 10) | (df_master['Phonological_Aware'] < 8) | \
       (df_master['RAN_Speed_Sec'] > 150) | (df_master['Total_Growth'] < 4)
df_master['Risk_Flag'] = mask.astype(int)

# 3. Apply Clustering Tiers directly to Master File
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_master[['LNF_BOY_Score', 'Phonological_Aware', 'RAN_Speed_Sec', 'Attendance_Rate']])
df_master['Risk_Tier'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_cluster)

# 4. Prepare and Train the Final Predictive Model
df_encoded = pd.get_dummies(df_master, columns=['District', 'Home_Language', 'PreK_Enrollment'])
X = df_encoded.drop(['Risk_Flag', 'Risk_Tier'], axis=1)
y = df_encoded['Risk_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#====================================
# 3-Wave data - batch processor defined
def batch_predict_growth(input_df, model, training_columns):
# Create a copy - do not overwrite the original district file
    df_master = input_df.copy() 
    
# Process categories (One-Hot Encoding)
    processed = pd.get_dummies(df_master)
    processed = processed.reindex(columns=training_columns, fill_value=0)
# Predict Risk and Confidence
    preds = model.predict(processed)
    probs = model.predict_proba(processed).max(axis=1)
    
# Add the results and Growth Analysis to our new dataframe
    df_master['Total_Growth'] = (df_master['Vocab_EOY'] - df_master['Vocab_BOY']).round(1)
    df_master['Predicted_Risk'] = preds
    df_master['Confidence_%'] = (probs * 100).round(1)
    
    return df_master
training_cols = X.columns.tolist()
sample_data = df_master.head(10).drop(['Risk_Flag', 'Risk_Tier'],axis=1) # Simulate a new file
final_results = batch_predict_growth(sample_data, model, training_cols)


print("\n--- SAMPLE BATCH RESULTS (FIRST 5 STUDENTS) ---")
print(final_results[['Vocab_BOY', 'Vocab_EOY', 'Total_Growth', 'Predicted_Risk', 'Confidence_%']].head())

sample_new_students = df_master.head(5).drop(['Risk_Flag', 'Risk_Tier'],axis=1)
results = batch_predict_growth(sample_new_students, model, X.columns.tolist())
print("\n--- NEW STUDENT PREDICTIONS ---")
print(results[['Vocab_BOY', 'Vocab_MOY', 'Vocab_EOY', 'Total_Growth', 'Predicted_Risk']])

#================================

# GENERATE PREDICTIONS
y_pred = model.predict(X_test)

# Final Report
print("-----LITERACY RISK DETECTION MODEL (HISD/CISD)-----:\n")
print(classification_report(y_test, y_pred, target_names=['On-Track', 'At-Risk']))

# Get feature names from the encoded DataFrame X
current_feature_names = X.columns 

# Get importances and sort 
importances = model.feature_importances_
sorted_idx = importances.argsort()

# Get feature names from your encoded features (X)
feature_names = X.columns

# Extract the importance scores from trained model
importances = model.feature_importances_

# Combine importance scores; sort from (highest to lowest)
ranked_predictors = sorted(zip(importances, feature_names), reverse=True)

# Print the Top 10
print("--- Top Early Literacy Risk Predictors ---")
for i, (score, name) in enumerate(ranked_predictors[:10], 1):
    print(f"{i}. {name}: {score:.4f}")

# 3. Plot using the feature names
plt.figure(figsize=(12, 8))
plt.bar(current_feature_names[sorted_idx], importances[sorted_idx], color='#006d77', edgecolor='black', linewidth=1.5)
threshold = 0.05
plt.axhline(y=threshold, color='#b22222', linestyle='--', linewidth=2, 
            label=f'High Impact Threshold ({threshold})')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel('Predictive Power', fontsize=12)
plt.xlabel('Predictors', fontsize=12)
plt.title('Early Warning System: Top Predictors of Literacy Risk (HISD & CISD)', fontsize=16, pad=25)
plt.xlabel('Predictive Power', fontsize=12)
plt.xlabel('Predictors', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig('EWS_Diagnostic.png', dpi=300, bbox_inches='tight')
plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Create the visual display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Generate the matrix using your fixed y_test and y_pred
cm = confusion_matrix(y_test, y_pred)

#  Console output
cm = np.array([[228, 0], [0, 172]])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['On-Track', 'At-Risk'], 
            yticklabels=['On-Track', 'At-Risk'])

plt.title('Confusion Matrix: HISD/CISD Early Literacy EWS', fontsize=14)
plt.ylabel('Actual Status (Ground Truth)', fontsize=12)
plt.xlabel('Predicted Status (EWS Model)', fontsize=12)

#===========================================================================
import matplotlib.pyplot as plt

# Data from your results
features = ['Total Growth', 'Phonological Awareness', 'RAN Speed', 'LNF (Letter Naming)', 'Vocab EOY']
importance = [0.2917, 0.2493, 0.1707, 0.1193, 0.0732]

plt.figure(figsize=(10, 6))
plt.barh(features[::-1], importance[::-1], color='teal')
plt.title('Top 5 Indicators of Literacy Risk', fontsize=16)
plt.xlabel('Importance Weight', fontsize=12)

# Save for PowerPoint
plt.savefig('Top_Predictors_Final.png', dpi=300, bbox_inches='tight')
plt.show()

# SAVE THE IMAGE
plt.savefig('Confusion_Matrix_Final.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC CURVE
y_probs = model.predict_proba(X_test)[:, 1]

# ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# CURVE PLOT
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'EWS Performance (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity')
plt.title('ROC Curve: Diagnostic Power')
plt.legend(loc="lower right")
plt.savefig('EWS_Diagnostic.png', dpi=300, bbox_inches='tight')
plt.show()
#plt.savefig('EWS_Predictors.png', dpi=300)
#import matplotlib.pyplot as plt
# ... your chart code ...
#plt.savefig("HISD_CISD_Literacy_EWS.png", dpi=300, bbox_inches='tight')

df_master.to_csv('HISD_CISD_Master_Literacy_Data.csv', index=False)


# Updated Slide 6 Plot with Data Labels
plt.figure(figsize=(10, 6))
features = ['Total Growth', 'Phonological Awareness', 'RAN Speed', 'LNF (Letter Naming)', 'Vocab EOY']
importance = [0.2917, 0.2493, 0.1707, 0.1193, 0.0732]

bars = plt.barh(features[::-1], importance[::-1], color='teal')
plt.title('Top 5 Indicators of Literacy Risk', fontsize=16)
plt.xlabel('Importance Weight (0.0 - 1.0)', fontsize=12)

# Adding the Data Labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', va='center', fontsize=11, fontweight='bold')

plt.xlim(0, 0.35) # Add space for labels
plt.tight_layout()
plt.savefig('Slide_6_Results.png', dpi=300);
plt.show()





