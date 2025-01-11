# Kredi Temerrüt Tahmini için Açıklanabilir Yapay Zeka (XAI)

Bu proje, kredi temerrüt tahminleri için Açıklanabilir Yapay Zeka (XAI) tekniklerini kullanarak bir model oluşturur ve sonuçlarını açıklar.

## Proje Yapısı

```
xai_credit_project/
├── data/               # Veri dosyaları
├── scripts/            # Python kodları
├── results/            # Model sonuçları ve görselleştirmeler
└── notebooks/          # Jupyter not defterleri
```

## Veri Seti

Veri seti (`Default_Fin.csv`) aşağıdaki özellikleri içerir:
- Employed: İstihdam durumu (1 = Evet, 0 = Hayır)
- Bank_Balance: Banka hesap bakiyesi
- Annual_Salary: Yıllık maaş
- Defaulted: Hedef değişken (1 = Temerrüt var, 0 = Temerrüt yok)

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Scripti çalıştırın:
```bash
cd scripts
python xai_credit_model.py
```

## Çıktılar

Script çalıştırıldığında, `results/` klasöründe aşağıdaki dosyalar oluşturulur:
- `model_performans.txt`: Sınıflandırma metrikleri
- `shap_ozet.png`: SHAP değerleri özet grafiği
- `lime_aciklama.html`: LIME açıklamaları

## Model Açıklamaları

- **SHAP (SHapley Additive exPlanations)**: Her bir özelliğin model tahminlerine olan katkısını gösterir
- **LIME (Local Interpretable Model-agnostic Explanations)**: Bireysel tahminler için yerel açıklamalar sağlar

## Gereksinimler

Gerekli Python paketleri `requirements.txt` dosyasında listelenmiştir. 