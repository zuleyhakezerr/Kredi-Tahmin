from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from scripts.xai_credit_model import *
import os
import traceback
import sys

app = Flask(__name__)

# CORS politikasını ayarla
CORS(app)

# Global değişkenler
model = None
X_train = None
X_test = None
scaler = None

@app.route('/api/train', methods=['POST'])
def train_model():
    global model, X_train, X_test, scaler
    
    try:
        print("Model eğitimi başlıyor...")
        
        # Veri yükleme kontrolü
        try:
            X_scaled, y, df, X_orig = veri_yukle_ve_onisle()
            print("Veri başarıyla yüklendi")
        except Exception as e:
            print(f"Veri yükleme hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"Veri yükleme hatası: {str(e)}")
        
        # Model eğitimi kontrolü
        try:
            model, X_train, X_test, y_train, y_test = model_egit(X_scaled, y)
            print("Model başarıyla eğitildi")
        except Exception as e:
            print(f"Model eğitim hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"Model eğitim hatası: {str(e)}")
        
        # Performans değerlendirme kontrolü
        try:
            performans_metrikleri, y_pred = model_performansi(model, X_test, y_test)
            print("Performans metrikleri hesaplandı")
        except Exception as e:
            print(f"Performans hesaplama hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"Performans hesaplama hatası: {str(e)}")
        
        # SHAP ve LIME açıklamaları
        try:
            shap_aciklamalar(model, X_train, X_test)
            lime_aciklamalar(model, X_train, X_test)
            print("Açıklamalar oluşturuldu")
        except Exception as e:
            print(f"Açıklama oluşturma hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"Açıklama oluşturma hatası: {str(e)}")
        
        # Özellik önem dereceleri
        feature_importance = pd.DataFrame({
            'özellik': X_orig.columns,
            'önem': model.feature_importances_
        }).sort_values('önem', ascending=False)
        
        # AI yorumu
        try:
            ai_interpretation = get_ai_interpretation(performans_metrikleri, feature_importance)
            print("AI yorumu alındı")
        except Exception as e:
            print(f"AI yorum hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"AI yorum hatası: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': 'Model başarıyla eğitildi',
            'performans': {
                'roc_auc': float(performans_metrikleri['roc_auc']),
                'classification_report': performans_metrikleri['classification_report']
            },
            'ai_interpretation': ai_interpretation
        })
        
    except Exception as e:
        error_msg = f"Model eğitimi sırasında hata oluştu: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'success': False,
            'message': str(e),
            'error_details': error_msg
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, X_train
    
    try:
        print("Tahmin işlemi başlıyor...")
        
        # Model kontrolü
        if model is None:
            return jsonify({
                'success': False,
                'message': 'Lütfen önce modeli eğitin!'
            }), 400
        
        data = request.get_json()
        print(f"Gelen veri: {data}")
        
        # Veri doğrulama
        required_fields = ['employed', 'bank_balance', 'annual_salary']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'Eksik alan: {field}'
                }), 400
        
        # Giriş verilerini hazırla
        try:
            input_data = pd.DataFrame({
                'Employed': [float(data['employed'])],
                'Bank Balance': [float(data['bank_balance'])],
                'Annual Salary': [float(data['annual_salary'])]
            })
            print("Giriş verileri hazırlandı")
        except ValueError as e:
            return jsonify({
                'success': False,
                'message': f'Geçersiz veri formatı: {str(e)}'
            }), 400
        
        # Tahmin yap
        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            print(f"Tahmin sonucu: {prediction}, Olasılıklar: {proba}")
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"Tahmin hatası: {str(e)}")
        
        # LIME açıklaması
        try:
            feature_names = ['İstihdam Durumu', 'Banka Bakiyesi', 'Yıllık Maaş']
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['Temerrüt Yok', 'Temerrüt Var'],
                mode='classification'
            )
            
            exp = explainer.explain_instance(
                input_data.iloc[0].values, 
                model.predict_proba,
                num_features=3
            )
            print("LIME açıklaması oluşturuldu")
        except Exception as e:
            print(f"LIME açıklama hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            raise Exception(f"LIME açıklama hatası: {str(e)}")
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': {
                'no_default': float(proba[0]),
                'default': float(proba[1])
            },
            'explanation': exp.as_list()
        })
        
    except Exception as e:
        error_msg = f"Tahmin sırasında hata oluştu: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'success': False,
            'message': str(e),
            'error_details': error_msg
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 