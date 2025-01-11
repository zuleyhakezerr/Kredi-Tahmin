#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kredi Temerrüt Tahmini için Açıklanabilir Yapay Zeka (XAI) Modeli
Bu script, kredi temerrüt tahminleri için RandomForest modeli oluşturur ve
SHAP ve LIME kullanarak model açıklamaları sağlar.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
import seaborn as sns
import json
import openai
from dotenv import load_dotenv

# Çevresel değişkenleri yükle
load_dotenv()

# OpenAI API anahtarını çevresel değişkenden al
openai.api_key = os.getenv('OPENAI_API_KEY')

# Sonuçlar klasörünü kontrol et ve oluştur
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
print(f"Sonuçlar dizini: {RESULTS_DIR}")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print("Sonuçlar dizini oluşturuldu")

def veri_yukle_ve_onisle():
    """Veri setini yükle, temizle ve ön işle"""
    try:
        # Veriyi yükle
        veri_yolu = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Default_Fin.csv")
        print(f"Veri yükleniyor: {veri_yolu}")
        
        if not os.path.exists(veri_yolu):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {veri_yolu}")
            
        df = pd.read_csv(veri_yolu)
        print(f"Veri yüklendi. Boyut: {df.shape}")
        
        # Eksik değerleri kontrol et ve temizle
        eksik_sayisi = df.isnull().sum().sum()
        print(f"Eksik değer sayısı: {eksik_sayisi}")
        df = df.dropna()
        print(f"Eksik değerler temizlendi. Yeni boyut: {df.shape}")
        
        # Aykırı değerleri tespit et ve temizle
        for col in ['Bank Balance', 'Annual Salary']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            print(f"{col} için aykırı değerler temizlendi. Yeni boyut: {df.shape}")
        
        # Özellikleri ve hedef değişkeni ayır
        X = df[['Employed', 'Bank Balance', 'Annual Salary']]
        y = df['Defaulted?']
        print("Özellikler ve hedef değişken ayrıldı")
        
        # Sayısal özellikleri ölçeklendir
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[['Bank Balance', 'Annual Salary']] = scaler.fit_transform(X[['Bank Balance', 'Annual Salary']])
        print("Özellikler ölçeklendirildi")
        
        return X_scaled, y, df, X
        
    except Exception as e:
        print(f"Veri yükleme ve ön işleme sırasında hata: {str(e)}")
        raise

def model_egit(X, y):
    """RandomForest modelini eğit ve optimize et"""
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE ile dengesiz veri setini dengele
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Model parametrelerini optimize et
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    return model, X_train_balanced, X_test, y_train_balanced, y_test

def model_performansi(model, X_test, y_test):
    """Model performansını detaylı olarak değerlendir"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Sınıflandırma raporu
    rapor = classification_report(y_test, y_pred)
    
    # ROC AUC skoru
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    performans_metrikleri = {
        'classification_report': rapor,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    with open(f"{RESULTS_DIR}/model_performans.txt", "w", encoding="utf-8") as f:
        f.write("Model Performans Raporu\n")
        f.write("=======================\n\n")
        f.write(f"ROC AUC Skoru: {roc_auc:.4f}\n\n")
        f.write(rapor)
    
    return performans_metrikleri, y_pred

def get_ai_interpretation(performans_metrikleri, feature_importance):
    """OpenAI API kullanarak model sonuçlarını yorumla"""
    prompt = f"""
    Kredi temerrüt tahmin modelinin sonuçlarını yorumla:
    
    1. Model Performans Metrikleri:
    - ROC AUC Skoru: {performans_metrikleri['roc_auc']}
    - Sınıflandırma Raporu:
    {performans_metrikleri['classification_report']}
    
    2. Özellik Önem Sıralaması:
    {feature_importance.to_string()}
    
    Lütfen şu konularda yorum yap:
    1. Modelin genel performansı nasıl?
    2. Hangi özellikler temerrüt tahmininde en etkili?
    3. Modelin güçlü ve zayıf yönleri neler?
    4. Model nasıl iyileştirilebilir?
    
    Yorumunu Türkçe olarak, maddeler halinde ve anlaşılır bir dille yap.
    """
    
    client = openai.OpenAI(
        api_key="..."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Sen bir finans ve makine öğrenmesi uzmanısın."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def create_html_report(model, X_orig, X_scaled, y, df, performans_metrikleri, y_test, y_pred, ai_interpretation):
    """Gelişmiş HTML raporu oluştur"""
    # Karmaşıklık matrisi
    cm_base64 = create_confusion_matrix(y_test, y_pred)
    
    # Özellik önem grafiği
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'özellik': X_orig.columns,
        'önem': model.feature_importances_
    }).sort_values('önem', ascending=False)
    
    sns.barplot(x='önem', y='özellik', data=feature_importance)
    plt.title('Özellik Önem Dereceleri')
    plt.tight_layout()
    importance_base64 = plot_to_base64(plt)
    plt.close()
    
    # Veri dağılımı grafikleri
    dist_plots = create_distribution_plots(X_orig, y)
    
    # Veri seti istatistikleri
    stats = df.describe().round(2).to_html(classes='table table-striped')
    
    # HTML�ablonu
    html_template = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kredi Temerrüt Tahmin Raporu</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; background-color: #f8f9fa; }
            .container { max-width: 1200px; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .section { margin-bottom: 40px; }
            img { max-width: 100%; height: auto; border-radius: 5px; }
            .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .interpretation { background-color: #e9ecef; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Kredi Temerrüt Tahmin Analizi</h1>
            
            <div class="section">
                <h2>Model Performansı</h2>
                <div class="row">
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4>ROC AUC Skoru</h4>
                            <h2>{roc_auc:.4f}</h2>
                        </div>
                    </div>
                </div>
                <pre>{classification_report}</pre>
            </div>
            
            <div class="section">
                <h2>AI Yorumu</h2>
                <div class="interpretation">
                    {ai_interpretation}
                </div>
            </div>
            
            <div class="section">
                <h2>Karmaşıklık Matrisi</h2>
                <img src="data:image/png;base64,{cm_base64}" alt="Karmaşıklık Matrisi">
            </div>
            
            <div class="section">
                <h2>Özellik Önem Dereceleri</h2>
                <img src="data:image/png;base64,{importance_base64}" alt="Özellik Önem Dereceleri">
            </div>
            
            <div class="section">
                <h2>Özellik Dağılımları</h2>
                <div class="row">
                    {dist_plots}
                </div>
            </div>
            
            <div class="section">
                <h2>Veri Seti İstatistikleri</h2>
                {stats}
            </div>
            
            <div class="section">
                <h2>Model Açıklamaları</h2>
                <p>Detaylı SHAP ve LIME açıklamaları için lütfen aşağıdaki dosyalara bakın:</p>
                <ul>
                    <li><a href="shap_ozet.png">SHAP Özet Grafiği</a></li>
                    <li><a href="lime_aciklama.html">LIME Açıklamaları</a></li>
                </ul>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # HTML içeriğini oluştur
    html_content = html_template.format(
        roc_auc=performans_metrikleri['roc_auc'],
        classification_report=performans_metrikleri['classification_report'],
        ai_interpretation=ai_interpretation.replace('\n', '<br>'),
        cm_base64=cm_base64,
        importance_base64=importance_base64,
        dist_plots=dist_plots,
        stats=stats
    )
    
    # HTML dosyasını kaydet
    with open(f"{RESULTS_DIR}/rapor.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def create_distribution_plots(X, y):
    """Her özellik için dağılım grafiklerini oluştur"""
    plots_html = ""
    
    for col in X.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=X, x=col, hue=y, multiple="stack")
        plt.title(f'{col} Dağılımı')
        plt.tight_layout()
        plot_base64 = plot_to_base64(plt)
        plt.close()
        
        plots_html += f"""
        <div class="col-md-6 mb-4">
            <img src="data:image/png;base64,{plot_base64}" alt="{col} Dağılımı">
        </div>
        """
    
    return plots_html

def plot_to_base64(plt):
    """Matplotlib grafiğini base64 formatına dönüştür"""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_confusion_matrix(y_test, y_pred):
    """Karmaşıklık matrisini oluştur"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    
    # Base64'e dönüştür
    cm_base64 = plot_to_base64(plt)
    plt.close()
    return cm_base64

def shap_aciklamalar(model, X_train, X_test):
    """SHAP değerlerini hesapla ve görselleştir"""
    try:
        print("SHAP açıklamaları oluşturuluyor...")
        
        # SHAP değerlerini hesapla
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Pozitif sınıf için SHAP değerleri
        
        print(f"SHAP değerleri hesaplandı. Boyut: {shap_values.shape}")
        
        # SHAP özet grafiği
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("SHAP Özellik Önem Dereceleri")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/shap_ozet.png")
        plt.close()
        print("SHAP özet grafiği kaydedildi")
        
        return shap_values
        
    except Exception as e:
        print(f"SHAP açıklamaları oluşturulurken hata: {str(e)}")
        raise

def lime_aciklamalar(model, X_train, X_test):
    """LIME açıklamaları oluştur"""
    try:
        print("LIME açıklamaları oluşturuluyor...")
        
        # LIME açıklayıcı oluştur
        feature_names = ['İstihdam Durumu', 'Banka Bakiyesi', 'Yıllık Maaş']
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['Temerrüt Yok', 'Temerrüt Var'],
            mode='classification'
        )
        
        # Örnek bir tahmin için LIME açıklaması
        exp = explainer.explain_instance(
            X_test.iloc[0].values, 
            model.predict_proba,
            num_features=3
        )
        
        # HTML olarak kaydet
        exp.save_to_file(f"{RESULTS_DIR}/lime_aciklama.html")
        print("LIME açıklaması HTML olarak kaydedildi")
        
    except Exception as e:
        print(f"LIME açıklamaları oluşturulurken hata: {str(e)}")
        raise

def main():
    """Ana fonksiyon"""
    print("Veri yükleniyor ve ön işleniyor...")
    X_scaled, y, df, X_orig = veri_yukle_ve_onisle()
    
    print("Model eğitiliyor...")
    model, X_train, X_test, y_train, y_test = model_egit(X_scaled, y)
    
    print("Model performansı değerlendiriliyor...")
    performans_metrikleri, y_pred = model_performansi(model, X_test, y_test)
    
    print("SHAP açıklamaları oluşturuluyor...")
    shap_aciklamalar(model, X_train, X_test)
    
    print("LIME açıklamaları oluşturuluyor...")
    lime_aciklamalar(model, X_train, X_test)
    
    print("AI yorumu alınıyor...")
    feature_importance = pd.DataFrame({
        'özellik': X_orig.columns,
        'önem': model.feature_importances_
    }).sort_values('önem', ascending=False)
    
    ai_interpretation = get_ai_interpretation(performans_metrikleri, feature_importance)
    
    print("HTML raporu oluşturuluyor...")
    create_html_report(model, X_orig, X_scaled, y, df, performans_metrikleri, y_test, y_pred, ai_interpretation)
    
    # HTML raporunun tam yolunu al
    rapor_yolu = os.path.abspath(f"{RESULTS_DIR}/rapor.html")
    
    print("\n" + "="*50)
    print("İşlem tamamlandı!")
    print("="*50)
    print(f"\nSonuçlar şu klasörde: {os.path.abspath(RESULTS_DIR)}")
    print(f"\nRaporu görüntülemek için tarayıcıda şu dosyayı açın:\n{rapor_yolu}")
    
    # HTML raporunu otomatik aç
    import webbrowser
    webbrowser.open('file://' + rapor_yolu)

if __name__ == "__main__":
    main() 