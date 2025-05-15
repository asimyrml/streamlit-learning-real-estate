# Yapay Zeka Etik Projesi - Emlak Fiyat Tahmini

## 📄 Proje Açıklaması
Bu proje, yapay zeka ile emlak fiyat tahmini yaparken aynı zamanda etik analiz (Fairness Metrics) gerçekleştirmeyi amaçlamaktadır. Python dili kullanılarak, DecisionTree, LinearRegression ve RandomForest modelleri ile çalışılmıştır.

## 🔧 Kurulum
1. Python 3.10 veya üzeri sürüm kurulu olmalıdır.
2. Gerekli kütüphaneleri yükleyin:
```
pip install -r requirements.txt
```
3. (Opsiyonel) Sanal ortam (`venv`) kullanmanız önerilir.

## 🚀 Uygulamanın Çalıştırılması
Streamlit arayüzünü başlatmak için:
```
streamlit run app.py
```

## ⚡ Kullanım
- Giriş bilgilerini (ilçe, oda sayısı, alan, yaş vb.) doldurun.
- Kullanmak istediğiniz modeli seçin.
- "Tahmin Yap" butonuna basarak sonuçları görün.

## 📊 Özellikler
- Veri Temizleme ve Aykırı Değer Ayıklama
- EDA (Veri Görselleştirme)
- Model Eğitimi ve Değerlendirme
- Etik Analiz (Demographic Parity, Equalized Odds, Predictive Parity)
- SHAP ile Model Açıklaması
- Canlı Web Arayüzü (Streamlit)

## 🧠 Etik Analiz Özeti
Modeller farklı demografik gruplar üzerinde adalet/fairness açısından değerlendirilmiştir. Tüm metrikler analiz edilmiştir.

## 👨‍💻 Kullanılan Teknolojiler
- Python 3.10
- Pandas, Numpy, Scikit-learn, SHAP, Streamlit

## 📚 Referanslar
Detaylı referans listesi ayrıca sağlanacaktır.
