import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = r"C:\pro++\house price prediction\Housing.csv"
MODEL_PATH = "house_price_model.pkl"
PIPELINE_PATH = "preprocessing_pipeline.pkl"

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    logging.info(f"Loaded dataset with shape {df.shape} and columns: {df.columns.tolist()}")
    return df

def preprocess_data(df, target='price'):
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    logging.info("Preprocessing complete.")
    return X_processed, y, preprocessor, numeric_cols, categorical_cols

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Evaluation Metrics -> RMSE: {rmse:.2f}, R²: {r2:.2f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return rmse, r2

def plot_feature_importance(model, numeric_cols, categorical_cols, preprocessor):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        cat_features = []
        if categorical_cols:
            cat_transformer = preprocessor.named_transformers_['cat']['onehot']
            cat_features = cat_transformer.get_feature_names_out(categorical_cols).tolist()

        feature_names = numeric_cols + cat_features

        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=sorted_features)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        logging.warning("Model does not support feature importance.")

def save_model(model, path):
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    logging.info(f"Preprocessing pipeline saved to {path}")

def main():
    df = load_data(DATA_PATH)

    X, y, preprocessing_pipeline, numeric_cols, categorical_cols = preprocess_data(df, target='price')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into train and test sets.")

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, numeric_cols, categorical_cols, preprocessing_pipeline)

    save_model(model, MODEL_PATH)
    save_pipeline(preprocessing_pipeline, PIPELINE_PATH)

if __name__ == "__main__":
    main()
