import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from learning import load_data, train_models, predict_price, calculate_ethics, explain_model_shap

st.set_page_config(page_title="🏠 Real Estate Price Prediction", layout="wide")

# Başlık
st.title("🏠 Real Estate Price Prediction with Ethics Evaluation")

# Veriyi yükle
df_clean = load_data()

# X ve y ayır
X = df_clean.drop(columns=['price'])
y = df_clean['price']

# Modelleri eğit
trained_models = train_models(X, y)

# Ana layout başlıyor
st.header("📥 Tahmin için Bilgilerinizi Girin")

with st.form("prediction_form"):
    cols = st.columns(4)

    district = cols[0].selectbox('District', df_clean['district'].unique())
    neighborhood = cols[1].selectbox('Neighborhood', df_clean['neighborhood'].unique())
    room = cols[2].number_input('Room Count', min_value=1, max_value=10, value=2)
    living_room = cols[3].number_input('Living Room Count', min_value=0, max_value=5, value=1)

    cols2 = st.columns(4)
    area = cols2[0].number_input('Area (m2)', min_value=10, max_value=1000, value=100)
    age = cols2[1].number_input('Building Age', min_value=0, max_value=100, value=5)
    floor = cols2[2].number_input('Floor', min_value=0, max_value=50, value=2)
    neighbourhoodScore = cols2[3].slider('Neighbourhood Score', 1, 10, 5)

    model_choice = st.selectbox('Model Seçimi', list(trained_models.keys()))
    submit = st.form_submit_button('📈 Tahmin Yap')

if submit:
    selected_model = trained_models[model_choice]
    input_data = pd.DataFrame({
        'district': [district],
        'neighborhood': [neighborhood],
        'room': [room],
        'living room': [living_room],
        'area (m2)': [area],
        'age': [age],
        'floor': [floor],
        'neighbourhoodScore': [neighbourhoodScore]
    })

    prediction = predict_price(selected_model, input_data)
    
    st.success(f"🏡 **Tahmin Edilen Fiyat:** {prediction:,.2f} ₺")

    st.divider()

    # Etik analiz
    st.subheader("⚖️ Etik Analiz Sonuçları")
    X_test = df_clean.drop(columns=['price'])
    y_test = df_clean['price']

    dp_diff, error_diff, eo_diff, pp_diff = calculate_ethics(selected_model, X_test, y_test)

    # 4'lü kart gösterimi
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if dp_diff < 100:
            st.success(f"Demographic Parity\n{dp_diff:.2f}")
        else:
            st.error(f"Demographic Parity\n{dp_diff:.2f}")

    with col2:
        if error_diff < 100:
            st.success(f"Error Parity\n{error_diff:.2f}")
        else:
            st.error(f"Error Parity\n{error_diff:.2f}")

    with col3:
        if eo_diff < 0.1:
            st.success(f"Equal Opportunity\n{eo_diff:.2f}")
        else:
            st.error(f"Equal Opportunity\n{eo_diff:.2f}")

    with col4:
        if pp_diff < 0.1:
            st.success(f"Predictive Parity\n{pp_diff:.2f}")
        else:
            st.error(f"Predictive Parity\n{pp_diff:.2f}")

    st.divider()

    # SHAP Açıklaması
    st.subheader("🔍 Özelliklerin Tahmine Etkisi (SHAP Analizi)")
    try:
        explain_model_shap(selected_model, X_test.sample(100, random_state=42))  # 100 örnekle
    except Exception as e:
        st.warning(f"SHAP açıklaması yapılamadı: {e}")
