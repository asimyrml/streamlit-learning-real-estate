# 🏠 Real Estate Price Prediction & Fairness Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-brightgreen)

> **Emlak fiyatlarını makine öğrenmesiyle tahmin eden, aynı zamanda adalet (fairness) metrikleri ve SHAP açıklamaları sunan Streamlit uygulaması.**

---

## İçindekiler
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Kullanım](#kullanım)
- [Veri ve Modeller](#veri-ve-modeller)
- [Etik Analiz](#etik-analiz)
- [Dosya Yapısı](#dosya-yapısı)


---

## Özellikler

- 🔍 **Tahmin** &nbsp;&nbsp;&nbsp;&nbsp;DecisionTree, RandomForest ve LinearRegression modelleri  
- ⚖️ **Fairness** &nbsp;&nbsp;&nbsp;Demographic Parity, Error Parity, Equal Opportunity, Predictive Parity  
- 🖼️ **SHAP** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Özellik etkilerini interaktif grafikle gösterir  
- 🌐 **Streamlit** &nbsp;Tek komutla web arayüzü  
- 🧹 **Veri Hazırlama** &nbsp;Aykırı değer temizleme, eksik değer doldurma, one-hot encoding  
- 🪄 **Önbellek** &nbsp;&nbsp;&nbsp;`@st.cache_*` ile eğitim ve veri yükleme hızlanır  

---

## Kurulum

```bash
git clone https://github.com/<kullanıcı-adı>/<repo-adı>.git
cd <repo-adı>
python -m venv venv              # opsiyonel
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Hızlı Başlangıç
```bash
streamlit run app.py
```
Tarayıcıda http://localhost:8501 açılır; forma bilgileri gir, modeli seç ve Tahmin Yap’a tıkla.

## Kullanım

| Alan                   | Açıklama                   |
| ---------------------- | -------------------------- |
| **district**           | İlçe (ör. Kadıköy)         |
| **neighborhood**       | Mahalle                    |
| **room**               | Oda sayısı                 |
| **living room**        | Salon sayısı               |
| **area (m2)**          | Net/brüt alan              |
| **age**                | Bina yaşı                  |
| **floor**              | Bulunduğu kat              |
| **neighbourhoodScore** | Sosyo-ekonomik skor (1-10) |

Tahmin sonrası dört kartta adalet metrikleri, ardından SHAP özet grafiği görüntülenir.

## Veri ve Modeller
Varsayılan veri kümesi my_dataset.csv’de. Kendi verinizi kullanacaksanız sütun adlarını koruyun.
| Model                     | Kısa Açıklama                  |
| ------------------------- | ------------------------------ |
| **DecisionTreeRegressor** | Basit ve yorumlanabilir        |
| **LinearRegression**      | Hızlı, düşük varyans           |
| **RandomForestRegressor** | Daha yüksek doğruluk (30 ağaç) |

## Etik Analiz
Aşağıdaki metrikler district sütunu hassas öznitelik kabul edilerek hesaplanır (eşik değerler app.py içinde):
- Demographic Parity (DP)
- Error Parity (EP)
- Equal Opportunity (EO)
- Predictive Parity (PP)

## Dosya Yapısı
```bash
├── app.py                  # Streamlit arayüzü
├── learning.py             # Learning main fonksiyonlar
│
├── my_dataset.csv          # Örnek veri
├── requirements.txt
└── README.md
```
---
**by Asım Yirmili**
