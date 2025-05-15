import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    df = pd.read_csv("my_dataset.csv")
    df['neighbourhoodScore'] = df['neighbourhoodScore'].fillna(df['neighbourhoodScore'].mode()[0])
    
    num_cols = ['room', 'living room', 'area (m2)', 'age', 'floor', 'price']
    Q1, Q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    
    return df_clean

def create_pipeline(num_features, cat_features, model):
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # DÜZENLENDİ
        ]), cat_features)
    ])
    
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('model', model)
    ])
    
    return pipeline

@st.cache_resource
def train_models(X, y):
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=30, random_state=42)
    }
    
    trained_models = {}
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for name, model in models.items():
        pipeline = create_pipeline(num_features, cat_features, model)
        pipeline.fit(X, y)
        trained_models[name] = pipeline
    
    return trained_models

def predict_price(model, input_data):
    prediction = model.predict(input_data)[0]
    return prediction

def calculate_ethics(model, X_test, y_test, sensitive_attr='district', threshold=0.2):
    grp = X_test.reset_index(drop=True)[sensitive_attr]
    preds = model.predict(X_test)
    actual = y_test.reset_index(drop=True)

    dp_diff = abs(pd.Series(preds).groupby(grp).mean().diff().abs().iloc[1])
    error_diff = abs(pd.Series(np.abs(actual - preds)).groupby(grp).mean().diff().abs().iloc[1])
    
    success = (np.abs(actual - preds) / actual) < threshold
    tpr_per_group = success.groupby(grp).mean()
    eo_diff = abs(tpr_per_group.diff().iloc[-1])
    
    pred_success = success.groupby(grp).mean()
    pp_diff = abs(pred_success.diff().iloc[-1])

    return dp_diff, error_diff, eo_diff, pp_diff

def explain_model_shap(model_pipeline, X_sample):

    # Pipeline içindeki adımlara eriş
    preprocessor = model_pipeline.named_steps['pre']
    model = model_pipeline.named_steps['model']

    # X_sample'ı preprocess yap
    X_transformed = preprocessor.transform(X_sample)

    # Özellik isimlerini oluştur
    num_features = preprocessor.named_transformers_['num'].feature_names_in_
    cat_features = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out()
    feature_names = list(num_features) + list(cat_features)

    # Doğru SHAP açıklayıcıyı kur
    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed, check_additivity=False)  # ← BURADA düzeltme
    elif isinstance(model, LinearRegression):
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)

    # SHAP summary plot çiz
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())
    plt.clf()
