# Walmart Sales Prediction with Random Forest

# Overview

This project applies a Random Forest Regressor to predict weekly sales using historical sales data from Walmart. The model is optimized using RandomizedSearchCV for hyperparameter tuning. Key steps include feature engineering, feature selection, model evaluation, and visualization.

# Dataset

The dataset includes:

1. Store: Unique identifier for each store.

2. Dept: Department number.

3. Date: The date of the observation.

3. Weekly_Sales: Target variable representing sales for the department.

4. Additional features: Year, Month, and Week extracted from the Date column.

# Features Engineering

1. Converted Date column to datetime format.

2. Extracted Year, Month, and Week from Date.

2. Dropped Date column after transformation.

3. Feature Importance Analysis to select the most significant features.

# Model Training

1. Used Random Forest Regressor with hyperparameter tuning via RandomizedSearchCV.

2. Evaluated model performance using MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² (R-Squared Score).

3. Predicted weekly sales for the test set.

# Visualizations

1. Feature Importance plot.

2. Actual vs Predicted Sales Scatter Plot.

3. Line Chart for Sales Trends.

# Requirements

To run the project, install the necessary dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

# Running the Code

Execute the script to train the model and generate predictions:

python walmart_sales_prediction.py

# Output

1. Walmart_Sales_Predictions.csv: Contains actual and predicted sales.

2. Visualizations for model evaluation.

# References
http://dataaspirant.com/2017/05/22/random-forest-algorithm-machine-learing/

https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

