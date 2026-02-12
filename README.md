
# Machine Learning Assignment 2  
## Breast Cancer Classification using Multiple ML Models  

---

## a) Problem Statement  

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset and deploy the final solution using Streamlit Community Cloud.

The task is to classify whether a breast tumor is **malignant (0)** or **benign (1)** using diagnostic medical features.

---

## b) Dataset Description  

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

- Source: UCI Machine Learning Repository  
- Number of Instances: 569  
- Number of Features: 30 numerical features  
- Target Variable: Binary Classification  
  - 0 → Malignant  
  - 1 → Benign  

The features are computed from digitized images of fine needle aspirate (FNA) of breast masses and describe characteristics such as radius, texture, perimeter, area, smoothness, concavity, etc.

The dataset satisfies the assignment requirements:
- ✔ More than 500 instances  
- ✔ More than 12 features  

---

## c) Models Used  

The following 6 classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## Evaluation Metrics Used  

For each model, the following metrics were calculated:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table  
![Models](images/table.png)


---

## Observations on Model Performance  

| ML Model | Observation |
|-----------|-------------|
| Logistic Regression | Performs very well due to good linear separability in the dataset. Provides stable and interpretable results. |
| Decision Tree | May slightly overfit the data but captures non-linear relationships effectively. |
| KNN | Performs well after feature scaling but is sensitive to distance metrics. |
| Naive Bayes | Fast and simple model; assumes feature independence which may slightly limit performance. |
| Random Forest | Provides strong performance by reducing overfitting through ensemble learning. |
| XGBoost | Achieves high performance due to gradient boosting and regularization, often outperforming single models. |

Overall, ensemble models (Random Forest and XGBoost) provide the most balanced and robust performance.

---

## Streamlit Application Features  

The deployed Streamlit application includes:

- CSV dataset upload option  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix  
- Classification report  

The application is deployed using **Streamlit Community Cloud**.

---

## Project Structure  
ML-Assignment-2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ │-- Logistic_Regression.pkl
│ │-- Decision_Tree.pkl
│ │-- KNN.pkl
│ │-- Naive_Bayes.pkl
│ │-- Random_Forest.pkl
│ │-- XGBoost.pkl
│ │-- scaler.pkl
│-- ML-Assignement-2.ipynb
|--test_file.csv

---

## Conclusion  

This project demonstrates the complete machine learning workflow including:

- Data preprocessing  
- Model implementation  
- Performance evaluation  
- Model comparison  
- Web application deployment  

The ensemble models achieved the best overall performance, highlighting the effectiveness of boosting and bagging techniques in classification tasks.
