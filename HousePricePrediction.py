#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:08:38 2024

@author: srilu (modified for Windows by ChatGPT)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

# ========= Load Dataset =========
# Make sure the CSV is in the same folder as this script
df = pd.read_csv("House Price dataset.csv")

# ========= Basic Info =========
print(df.head())
print(df.info())
print("Duplicates:", df.duplicated().sum())
print("Missing values (%):\n", round(df.isnull().sum() / df.shape[0] * 100, 2))
print("Unique values:\n", df.nunique())
print("Description:\n", round(df.describe(), 2))

# ========= Define Columns =========
num_cols = ['Sale_amount', 'Beds', 'Baths', 'Sqft_home', 'Sqft_lot', 'Age']
cat_cols = ['Type', 'Town', 'University']

# ========= Plot Histograms =========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, col in zip(axes.flatten(), num_cols):
    df[col].hist(ax=ax, edgecolor='black')
    ax.set_title(col)
plt.tight_layout()
plt.show()

# ========= Category Proportions =========
for col in cat_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts(normalize=True))
    print("*" * 40)

# ========= Correlation Heatmap =========
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="PuBu")
plt.title("Correlation Heatmap")
plt.show()

# ========= Preprocessing =========
df.drop(columns=['Record', 'Sale_date', 'Build_year'], inplace=True)
df['Type'] = df['Type'].replace({'Multi Family': 'Others', 'Multiple Occupancy': 'Others'})
df = pd.get_dummies(df, drop_first=True)

# ========= Feature & Target Split =========
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
fn = X.columns

# Z-score scaling (excluding dummies)
scaler = StandardScaler()
Xn = np.c_[scaler.fit_transform(X.iloc[:, :5]), X.iloc[:, 5:].values]

# ========= Split Data =========
Xn_train, Xn_test, y_train, y_test = train_test_split(Xn, y, test_size=0.3, random_state=1234)

# ========= Train Basic Model =========
dtr = DecisionTreeRegressor(random_state=1234)
dtr.fit(Xn_train, y_train)

# ========= Evaluate Basic Model =========
def evaluate_model(model, X_train, y_train, X_test, y_test, label=""):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"{label} R2 Score - Train: {r2_score(y_train, y_train_pred):.4f}")
    print(f"{label} MSE - Train: {mean_squared_error(y_train, y_train_pred):.2f}")
    print(f"{label} R2 Score - Test: {r2_score(y_test, y_test_pred):.4f}")
    print(f"{label} MSE - Test: {mean_squared_error(y_test, y_test_pred):.2f}")

evaluate_model(dtr, Xn_train, y_train, Xn_test, y_test, "Basic Decision Tree")

# ========= Hyperparameter Tuning =========
param_grid = {
    'max_depth': [5, 10, 15, 20, 30, 50],
    'min_samples_split': [5, 10, 15, 20, 30, 50],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': [10, 15, 20, 'auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=1234), param_grid,
                           cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(Xn_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Score (MSE):", -grid_search.best_score_)

# ========= Evaluate Tuned Model =========
evaluate_model(best_model, Xn_train, y_train, Xn_test, y_test, "Tuned Decision Tree")

# ========= Plot Tree =========
plt.figure(figsize=(25, 10))
plot_tree(best_model, max_depth=4, feature_names=fn, filled=True, fontsize=10)
plt.title("Decision Tree (Max Depth = 4)")
plt.show()

# ========= Feature Importances =========
importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
top_features = importance_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(9, 9))
sns.barplot(x=top_features.Importance, y=top_features.Feature)
plt.title("Top 15 Feature Importances")
for i, v in enumerate(top_features.Importance):
    plt.text(v + 0.002, i, f'{v*100:.2f}%', va='center')
plt.show()

# ========= Train with Top 15 Features =========
X_top = X[top_features.Feature]
dt_top = DecisionTreeRegressor()
dt_top.fit(X_top, y)

plt.figure(figsize=(25, 10))
plot_tree(dt_top, max_depth=4, filled=True, feature_names=top_features.Feature, fontsize=10)
plt.title("Decision Tree with Top 15 Features")
plt.show()
