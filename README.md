# Stroke Prediction Model
 # Stroke Prediction Using Logistic Regression

## Overview
This project focuses on predicting the likelihood of a stroke based on various demographic and health-related factors, such as age, gender, smoking habits, and BMI. Using machine learning techniques, specifically Logistic Regression, I aimed to identify individuals at a higher risk of experiencing a stroke. The process was done with data cleaning, exploration, and model evaluation to ensure reliable results.

## Dataset
The dataset contains 5110 entries with 12 features, including:
- **gender**: Male, Female, or Other
- **age**: Age of the individual
- **hypertension**: 0 for no hypertension, 1 if hypertension is present
- **heart_disease**: 0 for no heart disease, 1 if present
- **ever_married**: Yes or No
- **work_type**: Employment type
- **Residence_type**: Urban or Rural
- **avg_glucose_level**: Average blood glucose level
- **bmi**: Body Mass Index
- **smoking_status**: Smoking habits
- **stroke**: Target variable (0 = No stroke, 1 = Stroke)

### Dataset Highlights
- Total entries: 5110
- Features: 12
- Missing values:
  - BMI: 201 missing values (handled by imputing the mean)
- Imbalance: Only 4.87% of the data represents stroke cases, requiring us to address the imbalance with oversampling techniques.

## Tools and Libraries Used
- **Python Libraries:**
  - pandas, numpy: Data handling and analysis
  - seaborn, matplotlib: Data visualization
  - scikit-learn: Machine learning and preprocessing
  - imbalanced-learn: For oversampling using SMOTE

## Process
### 1. Data Preprocessing
- **Handling Missing Values:** Missing BMI values were replaced with the mean.
- **Dropping Irrelevant Features:** The `id` column was removed since it doesn’t contribute to predictions.
- **Encoding Categorical Variables:**
  - Converted features like `gender`, `ever_married`, and `Residence_type` into numerical values using Label Encoding.
  - Applied OneHotEncoding for multi-class variables like `work_type` and `smoking_status`.

### 2. Exploratory Data Analysis (EDA)
- Visualized age distributions and their relationship with strokes.
- Analyzed stroke proportions by gender and other features.
- Highlighted the imbalance in the dataset, showing the need for oversampling.

### 3. Balancing the Data
To address the class imbalance, we used SMOTE (Synthetic Minority Oversampling Technique). This created a balanced training set, ensuring equal representation of stroke and non-stroke cases.

### 4. Building the Model
- **Model:** Logistic Regression
- **Preprocessing:**
  - Standardized features using StandardScaler.
  - Split the data into training (80%) and testing (20%) sets.
- Trained the model and validated it using cross-validation.

### 5. Evaluating the Model
We assessed the model using the following metrics:
- **Cross-validation score:** 79.06%
- **Precision:** 16.49%
- **Recall:** 75.81%
- **ROC-AUC Score:** 75.51%
- **Confusion Matrix:** To evaluate true positives, false positives, true negatives, and false negatives.

## Key Results
The model performed moderately well, with an emphasis on recall. Since identifying stroke cases is critical, recall was prioritized to reduce false negatives.

## Visualizations
- Distribution of age for stroke vs. non-stroke cases.
- Count plots for gender, smoking status, and work type against stroke.
- ROC and Precision-Recall curves.

## Future Work
- Experiment with more advanced models like Random Forest, XGBoost, or Neural Networks.
- Include additional health-related features if available.
- Perform hyperparameter tuning to improve the model’s performance.
- Explore other data balancing methods for better results.

## How to Use This Project
1. **Install Required Libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
2. **Run the Script:**
   Make sure the dataset (`stroke.csv`) is in the same directory as the script. Execute the script to train the model and view results.
3. **View Visualizations:**
   The script generates helpful plots for understanding the data and evaluating the model.

## File Structure
```
project-folder/
|—— stroke.csv         # Dataset
|—— stroke_analysis.py # Python script
|—— README.md         # Documentation
```

## Conclusion
This project demonstrates how Logistic Regression can be used to predict health outcomes like strokes. With proper preprocessing and handling of class imbalance, the model achieved a balance between precision and recall. While there’s room for improvement, the project lays a solid foundation for further exploration in stroke prediction.


