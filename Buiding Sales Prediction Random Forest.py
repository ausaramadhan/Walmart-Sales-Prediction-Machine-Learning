#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')

#importing data
import os
df = pd.read_csv(r"D:\02 Kerjaan\Data\datasets\Walmart_Sales.csv")


# In[7]:


# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

# Drop rows with invalid dates (if any)
df = df.dropna(subset=["Date"])

# Feature Engineering: Extract Year, Month, Week from Date
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week


# In[10]:


# Drop original Date column (not needed for modeling)
df = df.drop(columns=["Date"])

# Define Features (X) and Target Variable (y)
X = df.drop(columns=["Weekly_Sales"])
y = df["Weekly_Sales"]

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize RandomizedSearchCV for optimization
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=10,  # Number of random combinations to try
    cv=3,       # 3-fold cross-validation
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit Randomized Search
random_search.fit(X_train, y_train)


# In[13]:


# Get Best Parameters and Best Model
best_params = random_search.best_params_
best_model = random_search.best_estimator_


# Initialize and Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions with Optimized Model
y_pred_opt = best_model.predict(X_test)

# Evaluate Optimized Model
mae_opt = mean_absolute_error(y_test, y_pred_opt)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
r2_opt = r2_score(y_test, y_pred_opt)

(mae_opt, rmse_opt, r2_opt)

# Print Results
print("Best Parameters:", best_params)
print("Mean Absolute Error (MAE):", mae_opt)
print("Root Mean Squared Error (RMSE):", rmse_opt)
print("R-Squared (R²):", r2_opt)


# In[16]:


from sklearn.metrics import accuracy_score

accuracy = best_model.score(X_test, y_test)
print("Model accuracy score with regressor:", accuracy)


# In[23]:


# Find Feature Importance
feature_importances = best_model.feature_importances_
feature_names = X.columns

# Create DataFrame for visualization
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"])
plt.title("Important Feature in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Select top features (adjust threshold as needed)
top_features = feature_importance_df["Feature"][:5].tolist()
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Train Random Forest Model on Selected Features
rf_selected = RandomForestRegressor(random_state=42, **best_params)
rf_selected.fit(X_train_selected, y_train)
y_pred_selected = rf_selected.predict(X_test_selected)

# Evaluate Optimized Model
mae_opt = mean_absolute_error(y_test, y_pred_selected)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_selected))
r2_opt = r2_score(y_test, y_pred_selected)

# Print Results
print("Best Parameters:", best_params)
print("Mean Absolute Error (MAE):", mae_opt)
print("Root Mean Squared Error (RMSE):", rmse_opt)
print("R-Squared (R²):", r2_opt)

# Model Accuracy Score (Regressor)
accuracy = rf_selected.score(X_test_selected, y_test)
print("Model accuracy score with regressor:", accuracy)

# Generate Classification Report & Confusion Matrix (Assuming classification problem)
y_test_class = np.where(y_test > y_test.median(), 1, 0)
y_pred_class = np.where(y_pred_selected > y_test.median(), 1, 0)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
print(conf_matrix)

# Slice Confusion Matrix
tp = conf_matrix[1, 1]  # True Positives
fp = conf_matrix[0, 1]  # False Positives
fn = conf_matrix[1, 0]  # False Negatives
tn = conf_matrix[0, 0]  # True Negatives

print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Negatives:", tn)


# In[28]:


# Predict sales on the test set
y_test_pred = best_model.predict(X_test)

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    "Id": X_test.index,  # Assuming 'Id' is the index
    "Weekly_Sales": y_test_pred  # Predicted sales
})
# Save submission DataFrame to CSV file without index
submission_df.to_csv(r"D:\02 Kerjaan\Data\datasets\Walmart_Sales_Predictions.csv", index=False)


# In[29]:


# Visualization: Actual vs. Predicted Weekly Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')  # Diagonal reference line
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title("Actual vs. Predicted Weekly Sales")
plt.show()

# Line Plot for Trend Comparison
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:50], label="Actual Sales", marker='o')  # Display first 50 points
plt.plot(y_test_pred[:50], label="Predicted Sales", marker='s')
plt.xlabel("Sample Index")
plt.ylabel("Weekly Sales")
plt.title("Weekly Sales Prediction Comparison")
plt.legend()
plt.show()


# In[ ]:




