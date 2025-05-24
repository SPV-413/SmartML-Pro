# SmartML Pro: End-to-End Machine Learning with AI Assistant 

I built this app using Streamlit, which provides an end-to-end ML solution for data science professionals to perform classification and regression tasks. It supports loading data from MySQL or file upload, EDA, comprehensive data preprocessing, model training with hyperparameter tuning, evaluation, and features an integrated AI assistant powered by Azure OpenAI GPT-4.1 for help and troubleshooting. The workflows for MySQL and file upload are similar. This app is designed to save time and speed up experimentation.

## Workflow 

1 EDA

2 Data Preprocessing Techniques Used:

- Handling missing values (dropping nulls)

- Removing duplicate rows

- Remove unrequired columns

- Outlier detection and removal using Z-score

- Encoding categorical variables with Label Encoding

- Skewness detection and reduction via Yeo-Johnson PowerTransformer

- Scaling numeric features using StandardScaler

- Feature Selection via Correlation Heatmap

3 Machine Learning Models Included:

- Classification: Logistic Regression, Random Forest Classifier, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN) Classifier, Decision Tree Classifier, Gradient Boosting Classifier, XGBoost Classifier, Multi-layer Perceptron (MLP) Classifier. 

- Regression: Linear Regression, Random Forest Regressor, Support Vector Regressor (SVR), K-Nearest Neighbors (KNN) Regressor, Decision Tree Regressor, Gradient Boosting Regressor, XGBoost Regressor, Multi-layer Perceptron (MLP) Regressor.

4 Hyperparameter Tuning:

- GridSearchCV with 5-fold cross-validation. 

- Model-specific parameters like estimators, max depth, learning rate, kernels, neighbors, activation functions, and regularization.

5 Evaluation Metrics:

- Classification: Accuracy Score, Classification Report (Precision, Recall, F1-score), Confusion Matrix. 

- Regression: Mean Squared Error, R2 Score, Mean Absolute Error. 

## AI Assistant:

Integrated AI assistant using Azure OpenAI GPT-4.1 model. Accessible via sidebar for help, troubleshooting, and guidance. 

## Business Use Cases:

- Rapid prototyping of ML models on business datasets. 

- Handling imbalanced classification problems with SMOTE. 

- Automating data cleaning and preprocessing to boost model performance. 

- Interactive AI support to accelerate data science workflows.

## Future improvements
Adding NLP preprocessing, clustering, deep learning, advanced visualization, and deployment features to this app.

## Link
[https://spv-413.streamlit.app/
](https://smartml-pro.streamlit.app/)

## Contact
For any inquiries, reach out via GitHub or email.
