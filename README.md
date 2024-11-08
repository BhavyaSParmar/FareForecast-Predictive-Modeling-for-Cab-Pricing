# FareForecast-Predictive-Modeling-for-Cab-Pricing

This project aims to build a predictive model to estimate cab fares based on various factors like distance, time, weather conditions, and cab type. Using a structured approach, I applied data preprocessing, exploratory data analysis (EDA), feature selection, and machine learning modeling to uncover insights and improve prediction accuracy. The goal was to create a robust data pipeline and employ machine learning techniques that can help understand fare patterns and support real-time decision-making in a rideshare context.

The project encompasses:

Data Preprocessing: Cleaning, encoding, and transforming raw data to ensure consistency and quality for modeling.
Exploratory Data Analysis (EDA): Analyzing data trends, including fare distributions across different cab types, weather conditions, and times of day.
Feature Selection & Engineering: Using Recursive Feature Elimination (RFE) to identify key variables that impact fare prices, which optimizes the model's performance.
Model Training & Evaluation: Experimenting with multiple machine learning models such as Linear Regression, Decision Trees, Random Forests, and Gradient Boosting, and evaluating them on performance metrics like MAE, MSE, and RMSE.
Interpretability with SHAP: Leveraging SHAP values to interpret feature importance and ensure model transparency.

The Phases can be divided in the following parts:
1. Data Collection and Preprocessing
Imported required packages such as pandas, numpy, matplotlib, seaborn, and sklearn.
Loaded the dataset, examined its shape and structure, and verified data types using .info() and .describe().
Handled missing values, cleaned data, and ensured consistency in the dataset by transforming date columns and handling null values.
Exploratory Data Analysis (EDA)

2. Analyzed various features and patterns in the data:
Cab Type vs. Price: Compared fares based on different cab types (e.g., Black SUV, Lux Black XL, Shared, UberPool).
Weather Conditions vs. Price: Investigated fare variations based on weather conditions (e.g., clear day vs. rainy day).
Distance vs. Price: Assessed the relationship between distance and price.
Hour vs. Price: Analyzed cab fares based on time of day.
Visualized data distributions and identified potential trends and anomalies using stripplot, scatter, and bar plots.
Data Preparation

3. Label Encoding: Converted categorical features (e.g., cab type, source, destination) into numerical format for model compatibility.
Binning: Created bins for surge multipliers to categorize wait times and represent them as integers.
Handling Missing Values: Filled missing values with median values for certain columns to maintain consistency in the data.
Recursive Feature Elimination (RFE)

Used RFE to identify and retain the most important features impacting the target variable (cab fare price).
Assigned dependent (price) and independent (influencing attributes) variables and conducted feature importance scoring to identify key features.
Tested RFE with 15, 25, 40, and 56 features to evaluate the effect on model performance.
Feature Selection
Identified and removed less impactful features to optimize the dataset.
Reduced dimensionality by selecting the most relevant columns, simplifying the dataset for model training.
Modeling and Testing

4. Model Selection: Trained and tested four ML models:
Linear Regression
Decision Tree
Random Forest
Gradient Boosting Regressor
Model Evaluation: Measured each model’s performance using:
Cross-validation
Performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Visualizations like scatter plots to compare actual vs. predicted values.
Result Analysis
Compiled model accuracy scores and error metrics for all algorithms.
Visualized accuracy comparisons and performance matrices for each model.
Feature Importance Analysis
Used SHAP (SHapley Additive exPlanations) values to interpret feature importance and evaluate the influence of each feature on predictions.
Visualized SHAP values to identify features with the most impact on cab fare predictions.
Additional Technical Components
Data Processing & Model Training: Used libraries like pandas, numpy, and sklearn for data manipulation, modeling, and evaluation.
Data Normalization: Used MinMaxScaler for feature scaling.
Validation Techniques: Cross-validation and scoring metrics helped ensure the model’s reliability and robustness.
