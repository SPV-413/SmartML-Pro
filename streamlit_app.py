import streamlit as st
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, mean_squared_error,
    r2_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.over_sampling import SMOTE
from scipy import stats
from collections import Counter
import mysql.connector
from mysql.connector import Error

# --- XGBoost Import with Check ---
try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_installed = True
except ImportError:
    xgb_installed = False

# --------- Streamlit App Setup ---------
st.set_page_config(page_title="ML App for Classification & Regression", layout="wide")
st.title("🤖 SmartML Pro: End-to-End Machine Learning with AI Assistant 🤖")

# --- 0. Load Data from MySQL (Optional) ---
st.header("0. Load Data from MySQL Database (Optional)")
use_mysql = st.checkbox("Load dataset from MySQL database")

# --- SESSION STATE INIT (NEW) ---
if 'df' not in st.session_state:
    st.session_state['df'] = None

df = st.session_state['df']  # Always use session_state['df']

if use_mysql:
    mysql_host = st.text_input("MySQL Host", value="localhost")
    mysql_user = st.text_input("MySQL User", value="root")
    mysql_password = st.text_input("MySQL Password", type="password")
    mysql_database = st.text_input("MySQL Database Name")

    # Use session state for table list
    if "mysql_tables" not in st.session_state:
        st.session_state.mysql_tables = []

    if st.button("Connect & List Tables"):
        try:
            connection = mysql.connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database
            )
            if connection.is_connected():
                st.success("Connected to MySQL database successfully!")
                cursor = connection.cursor()
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                cursor.close()
                connection.close()
                if tables:
                    st.session_state.mysql_tables = tables
                else:
                    st.warning("No tables found in the database.")
            else:
                st.error("Failed to connect to the database.")
        except Error as e:
            st.error(f"Error connecting to MySQL: {e}")

    if st.session_state.mysql_tables:
        selected_table = st.selectbox("Select Table to Load", st.session_state.mysql_tables)
        if st.button("Load Data from Selected Table"):
            try:
                connection = mysql.connector.connect(
                    host=mysql_host,
                    user=mysql_user,
                    password=mysql_password,
                    database=mysql_database
                )
                if connection.is_connected():
                    query = f"SELECT * FROM `{selected_table}`"
                    df = pd.read_sql(query, connection)
                    st.session_state['df'] = df  # <--- SAVE TO SESSION STATE
                    st.success(f"Data loaded successfully from table: {selected_table}")
                    st.dataframe(df.head())
                    connection.close()
                else:
                    st.error("Failed to connect to the database.")
            except Error as e:
                st.error(f"Error loading data: {e}")

# --- 1. Upload File (if not loaded from MySQL) ---
if st.session_state['df'] is None:
    st.header("1. Upload Data File (CSV, Excel, JSON)")
    file = st.file_uploader("Choose your dataset", type=["csv", "xlsx", "xls", "json"])

    if file:
        ext = file.name.split(".")[-1]
        if ext == "csv":
            df = pd.read_csv(file)
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(file)
        elif ext == "json":
            df = pd.read_json(file)

        if df is not None:
            st.session_state['df'] = df  # <--- SAVE TO SESSION STATE
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(df.head())

# Always use df from session state
df = st.session_state['df']

# Proceed only if df is loaded
if df is not None:
    # --- Unwanted Feature Removal Section ---
    st.subheader("Unwanted Feature Removal")
    option_list = ["Select an option", "Keep all features", "Remove selected features"]
    decision = st.radio("Do you want to remove any unwanted features?", option_list, index=0, key="remove_decision")
    if decision == "Remove selected features":
        columns_to_remove = st.multiselect("Select columns to remove from analysis:", options=list(df.columns), key="feature_drop")
    elif decision == "Keep all features":
        columns_to_remove = []
    else:
        columns_to_remove = None

    st.subheader("2. Exploratory Data Analysis (EDA)")
    if st.checkbox("Run Automated EDA Report"):
        with st.spinner("Generating EDA report..."):
            profile = ProfileReport(df, title="EDA", explorative=False)
            st_profile_report(profile)


    # Add a button to trigger the cleaning process
    if st.button("🧹 Clean Data"):
        if decision == "Select an option":
            st.warning("Please select an option for unwanted feature removal.")
        else:
            # Remove unwanted features if the decision is to remove and features are selected
            if decision == "Remove selected features" and columns_to_remove:
                df = df.drop(columns=columns_to_remove, errors='ignore')
            st.session_state['df'] = df
            st.success("Starting data cleaning process...")

            # Display current data types
            st.subheader("Data Types")
            st.write(df.dtypes)

            # Inspect object columns; show a message if none found
            st.subheader("Inspect Object Columns")
            object_cols = df.select_dtypes(include='object').columns
            if len(object_cols) == 0:
                st.write("No categorical data found in the dataset.")
            else:
                for col in object_cols:
                    st.write(f"Unique values in column '{col}':")
                    st.write(df[col].unique())

            # --- Data Cleaning & Preprocessing ---
            st.subheader("3. Data Cleaning & Preprocessing")
            st.write("Checking for missing values...")
            df = df.dropna()
            st.write("Null values removed.")

            st.write("Removing duplicate rows...")
            df = df.drop_duplicates()
            st.write("Duplicate rows removed.")

            st.write("Detecting outliers using Z-score...")
            numerical_cols = df.select_dtypes(include=['float', 'int']).columns
            if len(numerical_cols) > 0:
                z_scores = np.abs(stats.zscore(df[numerical_cols]))
                df = df[(z_scores < 3).all(axis=1)]
                st.write("Outliers removed.")
            else:
                st.write("No numerical columns found for outlier detection.")

            st.write("Encoding categorical variables...")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                st.write("Categorical variables encoded.")
            else:
                st.write("No categorical columns found for encoding.")

            st.write("Checking skewness of numeric features...")
            skew_threshold = 0.75  # threshold for skewness to transform
            numerical_cols = df.select_dtypes(include=['float', 'int']).columns
            skewed_feats = df[numerical_cols].skew().abs()
            features_to_transform = skewed_feats[skewed_feats > skew_threshold].index.tolist()

            if features_to_transform:
                st.write(f"Features with high skewness (> {skew_threshold}): {features_to_transform}")
                pt = PowerTransformer(method='yeo-johnson')
                df[features_to_transform] = pt.fit_transform(df[features_to_transform])
                st.write("Applied Yeo-Johnson transformation to reduce skewness.")
            else:
                st.write("No highly skewed features detected.")

            st.write("Scaling numeric features...")
            numerical_cols_sel = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols_sel) > 0:
                scaler = StandardScaler()
                df[numerical_cols_sel] = scaler.fit_transform(df[numerical_cols_sel])
            st.write("Numeric features scaled.")

            st.session_state['df'] = df
            st.success("Data cleaning process completed.")
            st.session_state['cleaned'] = True

    # Proceed with further steps only after cleaning is completed
    if st.session_state.get('cleaned', False):
        if st.checkbox("Show Correlation Heatmap"):
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.subheader("4. Select Target and Features")
        target_col = st.selectbox("Select Target Column", df.columns)
        default_features = [col for col in df.columns if col != target_col]
        feature_cols = st.multiselect("Select Feature Columns", default_features, default=default_features)

        if not feature_cols:
            st.error("Please select at least one feature column.")
        else:
            X = df[feature_cols]
            y = df[target_col]

            st.subheader("5. Data Splitting")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            st.subheader("6. Model Training & Evaluation")
            task_type = st.radio("Select Task Type", ["Classification", "Regression"])

            # SMOTE imbalance check and option only for classification
            if task_type == "Classification":
                # Ensure y_train is integer for SMOTE
                if not np.issubdtype(y_train.dtype, np.integer):
                    y_train = LabelEncoder().fit_transform(y_train)
                    y_test = LabelEncoder().fit_transform(y_test)

                class_counts = Counter(y_train)
                st.write(f"Class distribution in training set: {class_counts}")

                if len(class_counts) < 2:
                    st.error("Classification requires at least two classes in the target variable.")
                else:
                    majority_class_count = max(class_counts.values())
                    minority_class_count = min(class_counts.values())
                    imbalance_ratio = minority_class_count / majority_class_count if majority_class_count > 0 else 0

                    if imbalance_ratio < 0.5:
                        smote_option = st.checkbox("Handle Class Imbalance with SMOTE")
                        if smote_option:
                            try:
                                if minority_class_count < 2:
                                    st.warning("SMOTE skipped: Not enough samples in the minority class (need at least 2).")
                                else:
                                    k_neighbors = min(5, minority_class_count - 1)
                                    if k_neighbors < 1:
                                        st.warning("SMOTE skipped: Not enough samples in the minority class after cleaning.")
                                    else:
                                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                        X_train, y_train = smote.fit_resample(X_train, y_train)
                                        st.success(f"Class imbalance handled using SMOTE (k_neighbors={k_neighbors}).")
                            except ValueError as ve:
                                st.error(f"SMOTE Error: {ve}")
                    else:
                        st.info("Target classes appear balanced; SMOTE is not necessary.")
            else:
                st.info("SMOTE is only applicable for Classification tasks.")

            # Model selection and training
            if feature_cols and ((task_type == "Regression") or (task_type == "Classification" and len(Counter(y_train)) > 1)):
                if task_type == "Classification":
                    model_options = [
                        "Logistic Regression", "Random Forest", "SVM", "KNN",
                        "Decision Tree", "Gradient Boosting", "XGBoost", "MLP"
                    ]
                    if not xgb_installed:
                        model_options.remove("XGBoost")
                    model_name = st.selectbox("Choose Classifier", model_options)
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                        params = {'C': [0.1, 1, 10]}
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                        params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
                    elif model_name == "SVM":
                        model = SVC(random_state=42)
                        params = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
                    elif model_name == "KNN":
                        model = KNeighborsClassifier()
                        params = {'n_neighbors': [3, 5, 7]}
                    elif model_name == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)
                        params = {'max_depth': [3, 5, 10], 'criterion': ['gini', 'entropy']}
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier(random_state=42)
                        params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
                    elif model_name == "XGBoost":
                        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
                    elif model_name == "MLP":
                        model = MLPClassifier(max_iter=500, random_state=42)
                        params = {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}
                else:
                    model_options = [
                        "Linear Regression", "Random Forest", "SVR", "KNN",
                        "Decision Tree", "Gradient Boosting", "XGBoost", "MLP"
                    ]
                    if not xgb_installed:
                        model_options.remove("XGBoost")
                    model_name = st.selectbox("Choose Regressor", model_options)
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                        params = {}
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(random_state=42)
                        params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
                    elif model_name == "SVR":
                        model = SVR()
                        params = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
                    elif model_name == "KNN":
                        model = KNeighborsRegressor()
                        params = {'n_neighbors': [3, 5, 7]}
                    elif model_name == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=42)
                        params = {'max_depth': [3, 5, 10], 'criterion': ['squared_error', 'friedman_mse']}
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(random_state=42)
                        params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
                    elif model_name == "XGBoost":
                        model = XGBRegressor(random_state=42)
                        params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
                    elif model_name == "MLP":
                        model = MLPRegressor(max_iter=500, random_state=42)
                        params = {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}

                if st.button("Train and Evaluate Model"):
                    try:
                        if len(X_train) < 5:
                            st.warning("Not enough data for 5-fold cross-validation. Using leave-one-out instead.")
                            cv = len(X_train)
                        else:
                            cv = 5
                        grid = GridSearchCV(model, params, cv=cv, scoring='accuracy' if task_type == 'Classification' else 'r2')
                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                        y_pred = best_model.predict(X_test)

                        st.write("Best Parameters:", grid.best_params_)
                        if task_type == "Classification":
                            acc = accuracy_score(y_test, y_pred)
                            st.write(f"Accuracy Score: {acc:.4f}")
                            st.text("Classification Report")
                            st.text(classification_report(y_test, y_pred))
                            st.write("Confusion Matrix")
                            st.write(confusion_matrix(y_test, y_pred))
                        else:
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write(f"Mean Squared Error: {mse:.4f}")
                            st.write(f"R2 Score: {r2:.4f}")
                            st.write(f"Mean Absolute Error: {mae:.4f}")
                    except ValueError as ve:
                        st.error(f"Model training error: {ve}")
            else:
                st.warning("Cannot proceed: Not enough classes for classification, or no features selected.")

# --- Azure Endpoint and AI Assistant Chat Setup ---
AZURE_ENDPOINT = "https://models.github.ai/inference"
AZURE_MODEL = "openai/gpt-4.1"

# Get your GitHub token from environment variable or Streamlit secrets
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")

# Initialize client once
def get_client():
    if not GITHUB_TOKEN:
        st.sidebar.error("GitHub token not found! Please set GITHUB_TOKEN in environment or Streamlit secrets.")
        return None
    return ChatCompletionsClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(GITHUB_TOKEN),
    )

# --- Sidebar UI ---
st.sidebar.header("🤖 AI Assistant Chat (GitHub GPT-4.1)")

chat_input = st.sidebar.text_area("Ask the AI assistant for help or troubleshooting:")

if st.sidebar.button("Send to AI Assistant"):
    if not chat_input.strip():
        st.sidebar.warning("Please enter a question or message.")
    else:
        client = get_client()
        if client:
            try:
                response = client.complete(
                    messages=[
                        SystemMessage("You are a helpful AI assistant."),
                        UserMessage(chat_input),
                    ],
                    temperature=1,
                    top_p=1,
                    model=AZURE_MODEL,
                )
                answer = response.choices[0].message.content
                st.sidebar.markdown("*AI Assistant Response:*")
                st.sidebar.markdown(answer)
            except Exception as e:
                st.sidebar.error(f"API Error: {e}")

# --- Footer: Copyright ---
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f0f2f6;
    color: #555555;
    text-align: center;
    padding: 8px 0;
    font-size: 14px;
    border-top: 1px solid #e6e6e6;
    z-index: 1000;
}
</style>
<div class="footer">
    © 2025 Peraisoodan Viswanath S. All rights reserved.
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)


