# Stream.py — simple ML dataset predictor (upload → EDA → baseline train → download)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# Page config and title
st.set_page_config(page_title="ML Dataset Predictor", layout="wide")
st.title("📊 ML Dataset Predictor")
st.markdown("Upload a CSV, inspect it, train a baseline model (classification/regression), and download the trained model.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV file to get started. Try small files first (e.g., iris, titanic).")
    st.stop()

# Read CSV into a DataFrame
df = pd.read_csv(uploaded_file)

# --- Quick preview & info ---
st.subheader("🔎 Dataset preview")
st.write(df.head())                     # show first rows
st.write("**Shape:**", df.shape)        # rows, columns
st.subheader("ℹ️ Info & stats")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())              # prints dtypes & non-null counts
st.write(df.describe(include='all').T)  # descriptive stats

# --- Missing values & column types ---
st.subheader("🧭 Missing values and column types")
st.write(df.isnull().sum().sort_values(ascending=False))
st.write("Column dtypes:", df.dtypes)

# --- Simple visualizations for numeric columns ---
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    st.subheader("📈 Numeric column distributions")
    col_to_plot = st.selectbox("Choose numeric column to visualize", num_cols)
    fig = px.histogram(df, x=col_to_plot, nbins=30, title=f"Distribution of {col_to_plot}")
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap (numeric-only)
if len(num_cols) >= 2:
    st.subheader("🔗 Correlation matrix (numeric columns)")
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Target selection and problem type detection ---
st.subheader("🎯 Select target (label) column for supervised learning")
target_col = st.selectbox("Target column (leave if you want only EDA)", options=[None] + list(df.columns))
if target_col is None:
    st.info("No target selected. Use EDA. Choose a target column to enable model training.")
    st.stop()

y = df[target_col]
# Heuristic to detect regression vs classification
if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
    problem_type = "regression"
else:
    problem_type = "classification"
st.write("Detected problem type:", problem_type)

# --- Preprocessing (simple, robust) ---
st.subheader("⚙️ Preprocessing options")
scale_features = st.checkbox("Scale numeric features (StandardScaler)", value=False)
test_size = st.slider("Test set proportion", 5, 50, 20) / 100.0

# Prepare X, y
X = df.drop(columns=[target_col]).copy()
y = y.copy()

# Impute numeric columns with median
num_features = X.select_dtypes(include=[np.number]).columns
if len(num_features) > 0:
    X[num_features] = X[num_features].fillna(X[num_features].median())

# Impute categorical columns with a placeholder
cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
if len(cat_features) > 0:
    X[cat_features] = X[cat_features].fillna("__MISSING__")

# One-hot encode categorical features (simple approach)
X = pd.get_dummies(X, drop_first=True)

# Encode target for classification if needed
label_encoder = None
if problem_type == "classification":
    if not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))
    else:
        # numeric labels but may be strings of numbers; keep numeric
        y = y.values

# Scale if requested
if scale_features:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# --- Model selection ---
st.subheader("🤖 Choose a baseline model")
if problem_type == "classification":
    model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest (Classifier)"])
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest (Regressor)"])
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train button
if st.button("Train model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.success("Model trained ✅")
    # --- Metrics & visuals ---
    if problem_type == "classification":
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision (weighted):** {precision:.4f}")
        st.write(f"**Recall (weighted):** {recall:.4f}")
        st.write(f"**F1 (weighted):** {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC AUC for binary if available
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            probs = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probs)
            st.write(f"**ROC AUC:** {roc:.4f}")

    else:  # regression
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        # Predicted vs actual
        fig_scatter = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Predicted vs Actual")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Allow model download ---
    model_package = {"model": model, "label_encoder": label_encoder}
    model_bytes = pickle.dumps(model_package)
    st.download_button("⬇️ Download trained model (pickle)", data=model_bytes, file_name="trained_model.pkl")
