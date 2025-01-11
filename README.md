# 🏦 Kredi Temerrüt Tahmini XAI Projesi

Bu proje, makine öğrenmesi kullanarak kredi temerrüt tahminleri yapar ve tahminleri Açıklanabilir Yapay Zeka (XAI) teknikleriyle yorumlar.

## 🌟 Özellikler

- 🤖 RandomForest modeli ile kredi temerrüt tahmini
- 📊 SHAP ve LIME ile model açıklamaları
- 🎯 Detaylı performans metrikleri ve görselleştirmeler
- 💻 Flask API backend
- ⚛️ React ve Vue.js frontend seçenekleri
- 🔍 OpenAI entegrasyonu ile otomatik model yorumları

## 🛠️ Teknolojiler

- **Backend:**
  - Python 3.8+
  - scikit-learn
  - SHAP
  - LIME
  - Flask
  - OpenAI API

- **Frontend:**
  - React.js
  - Vue.js
  - Material-UI
  - Chart.js

## 📋 Gereksinimler

- Python 3.8 veya üzeri
- Node.js 14 veya üzeri
- pip (Python paket yöneticisi)
- npm veya yarn (Node.js paket yöneticisi)
- OpenAI API anahtarı

## 🚀 Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/kredi-tahmin-xai.git
cd kredi-tahmin-xai
```

2. Python sanal ortamı oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
# veya
venv\\Scripts\\activate  # Windows için
```

3. Python bağımlılıklarını yükleyin:
```bash
pip install -r requirements.txt
```

4. Çevresel değişkenleri ayarlayın:
```bash
cp .env.example .env
# .env dosyasını düzenleyip OpenAI API anahtarınızı ekleyin
```

5. Frontend bağımlılıklarını yükleyin (React için):
```bash
cd frontend-react
npm install
```

## 🎮 Kullanım

1. Backend'i başlatın:
```bash
python api/app.py
```

2. Frontend'i başlatın (React):
```bash
cd frontend-react
npm run dev
```

3. Tarayıcınızda şu adresi açın: `http://localhost:3000`

## 📊 Model Performansı

Model performans metrikleri `results` klasöründe bulunabilir:
- `model_performans.txt`: Sınıflandırma metrikleri
- `shap_ozet.png`: SHAP değerleri özet grafiği
- `lime_aciklama.html`: LIME açıklamaları
- `rapor.html`: Detaylı model raporu

## 🔒 Güvenlik

- API anahtarları ve hassas bilgiler `.env` dosyasında saklanmalıdır
- `.env` dosyası `.gitignore`'a eklenmiştir
- Tüm API istekleri güvenlik kontrolleri içerir

## 📁 Proje Yapısı

```
xai_credit_project/
├── api/                # Flask API
├── data/              # Veri dosyaları
├── frontend/          # Vue.js frontend
├── frontend-react/    # React frontend
├── results/           # Model çıktıları
├── scripts/           # Python scriptleri
└── notebooks/         # Jupyter not defterleri
```


Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 İletişim
<img width="862" alt="Ekran Resmi 2025-01-11 17 40 22" src="https://github.com/user-attachments/assets/3d45f391-556a-4052-99bf-705df77a562f" />
<img width="547" alt="Ekran Resmi 2025-01-11 17 40 44" src="https://github.com/user-attachments/assets/de4344ef-94ba-4986-8bb5-2c1f9deb7f74" />


## ⭐ Teşekkürler

Bu projeyi beğendiyseniz ⭐️ vermeyi unutmayın!
