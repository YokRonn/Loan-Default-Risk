# **Predicting Loan Default Risk**
# **Project Overview**
Accurately predicting loan defaults is crucial for both lenders and borrowers. For lenders, it safeguards assets and ensures financial stability, while borrowers gain access to fair lending practices and credit under favorable terms.
This project explores the creation of a machine learning models aimed at predicting the risk of loan defaults.
# **Business Understanding**
Loan default prediction plays a vital role in the lending process, enabling lenders to evaluate the risk of borrowers failing to repay their loans. By examining factors such as income, credit history, Work Experience, and economic trends, this project aims to enable lenders decide on the best model to mitigate risks and reduce potential losses.
#### Data Understanding
Data Source: https://www.kaggle.com/datasets/yaminh/applicant-details-for-loan-approve
This dataset provides insights into various attributes of loan applicants in india, essential for assessing their eligibility for loan approval.
It includes details such as annual income, age, work experience, marital status and more. It is designed to support loan approval processes by offering insights into applicant profiles aiding in risk assessment and facilitating informed lending decisions.

Here's a brief explanation of each column:

    1-Applicant_ID: Unique identifier for each loan applicant.
    2-Annual_Income: Annual income of the loan applicant.
    3-Applicant_Age: Age of the loan applicant.
    4-Work_Experience: Number of years of work experience of the loan applicant.
    5-Marital_Status: Marital status of the loan applicant.
    6-House_Ownership: Ownership status of the applicant's residence.
    7-Vehicle_Ownership(car): Ownership status of the applicant's vehicle.
    8-Occupation: Profession or occupation of the loan applicant.
    9-Residence_City: City where the loan applicant resides.
    10-Residence_State: State where the loan applicant resides.
    11-Years_in_Current_Employment: Number of years the applicant has been in their current job.
    12-Years_in_Current_Residence: Number of years the applicant has been residing in their current residence.
    13-Loan_Default_Risk: Indicator of loan default risk, with values indicating whether the loan applicant is at risk of defaulting on the loan.Remove Duplicates
#### Analysis and Observations
![image](https://github.com/user-attachments/assets/b3e8f1c8-8dd4-401c-b1e4-6cb496731476)
# Observations and Interpretations
The heatmap represents matrix of numerical features in the dataset. The colors indicate the strength and direction between variables i.e Dark colors signify stronger correlations while lighter colors suggest weaker or no correlation

1. Diagonal Values (Correlation of 1.0):
   The diagonal elements represent the correlation of a variable with itself, so the value is always 1.0.

2. High Correlation (e.g., Annual_Income and Years_in_Current_Employment with 0.64):
   This indicates a moderate positive relationship. As "Years_in_Current_Employment" increases, "Annual_Income" tends to increase as well.
   
3. Weak Correlation:
   Most other variables show weak or no correlation (values close to 0), suggesting limited linear relationships among them.

4. Loan_Default_Risk Correlation:
   The "Loan_Default_Risk" variable shows low correlations with all other features, suggesting no strong linear relationship with any single predictor.
   This analysis implies that while some features may have moderate relationships with each other, no single variable has a significant linear correlation with loan default risk, emphasizing the importance of using more complex models for prediction.


![image](https://github.com/user-attachments/assets/2162923e-2375-4fb5-9bdb-50e6ca5a865b)

# Key Observations

Top Feature - Applicant_ID (Importance: 0.252702):
This feature doesn't represent meaningful data and should not be prioritized while building the model in production.

Significant Features:

Annual_Income (Importance: 0.070205):
Higher income may reduce default risk, reflecting financial stability.

Applicant_Age (Importance: 0.059605):
Older applicants might default less due to financial maturity or more stable employment.

Work_Experience (Importance: 0.047731):
More work experience can imply a stable income, reducing default risk.

Years_in_Current_Employment & Years_in_Current_Residence:
Longer durations in employment or residence likely indicate stability, lowering default probabilities.

Lower Importance Features:

Marital_Status_married & Marital_Status_single:
These have minimal influence, suggesting marital status may not be a major determinant of default risk in this dataset.

Residence_State (e.g., Uttar Pradesh, West Bengal):
These features have low importance, but they might capture some regional socio-economic patterns.

# Using Random Forest Classifier

Accuracy: 0.9337

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.97      0.96     17447
           1       0.75      0.72      0.73      2553

    accuracy                           0.93     20000
   macro avg       0.86      0.84      0.85     20000
weighted avg       0.93      0.93      0.93     20000

The Random Forest model shows good overall performance with an accuracy of 93.37%. Here's a summary of the key metrics:

Strengths:
Low-Risk Cases (Class 0):
High precision (96%), recall (97%), and F1-score (96), indicating excellent performance in identifying low-risk borrowers.

Weaknesses:
High-Risk Cases (Class 1):
Moderate precision (75%) and recall (72%), meaning the model misses about 28% of high-risk cases and misclassifies some low-risk cases as high-risk.

Implications:
The model excels at identifying low-risk borrowers, but improvements are needed for detecting high-risk cases, which are essential for minimizing loan defaults.

Recommendations:
Techniques like class balancing, oversampling to improve recall for high-risk borrowers.


# Using Decision Tree Classifier

Decision Tree Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.94      0.94     17447
           1       0.58      0.57      0.57      2553

    accuracy                           0.89     20000
   macro avg       0.76      0.75      0.76     20000
weighted avg       0.89      0.89      0.89     20000

Decision Tree Accuracy: 0.89

The Decision Tree model has an accuracy of 89.3%, with strong performance for low-risk borrowers but poor results for high-risk borrowers.

Strengths:
Low-risk cases (Class 0): High precision (94%) and recall (94%), showing excellent performance in identifying low-risk borrowers.

Weaknesses:
High-risk cases (Class 1): Low precision (58%) and recall (57%), meaning the model misses many high-risk borrowers and misclassifies some low-risk cases as high-risk.

Business Implications:
The model performs well for low-risk borrowers but has a significant gap in detecting high-risk borrowers, which could lead to missed opportunities for managing loan defaults.

Recommendation:
Techniques like class balancing, threshold adjustment to improve recall for high-risk cases.


# Using Logistic Regression Classifier

Logistic Regression Accuracy: 0.8724

Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.87      1.00      0.93     17447
           1       1.00      0.00      0.00      2553

    accuracy                           0.87     20000
   macro avg       0.94      0.50      0.47     20000
weighted avg       0.89      0.87      0.81     20000

The Logistic Regression model achieves 87.24% accuracy, performing well for low-risk cases but completely failing to detect high-risk cases.

Strengths:
Low-risk cases (Class 0): Perfect recall (100%) and strong overall performance.

Weaknesses:
High-risk cases (Class 1): Recall is 0%, meaning the model fails to identify any high-risk borrowers.

Business Implications:
While effective for identifying low-risk borrowers, the model's inability to detect high-risk cases poses a significant risk of undetected loan defaults.

Recommendations:
Class imbalance can be addressed with oversampling or class weighting.
Adjusting the classification threshold to improve sensitivity to high-risk cases.
Exploring advanced algorithms like Gradient Boosting for better performance.


#### Conclusion
Preffered Model: Random Forest
Given its high accuracy (0.93), strong recall for class 1 (0.72) and balanced performance across both classes, Random Forest should be chosen as the primary model for predicting loan default risk.
It provides the best trade off between precision and recall, especially for the minority class (1).

By prioritizing Random Forest while addressing class imbalance, the solution will be the robust, accurate and better suited for identifying loan defaults.
