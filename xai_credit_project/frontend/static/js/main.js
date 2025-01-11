document.addEventListener('DOMContentLoaded', function() {
    const trainButton = document.getElementById('trainModel');
    const predictionForm = document.getElementById('predictionForm');
    const trainingStatus = document.getElementById('trainingStatus');
    const modelPerformance = document.getElementById('modelPerformance');
    const predictionResult = document.getElementById('predictionResult');

    // Model Eğitimi
    trainButton.addEventListener('click', async function() {
        try {
            trainButton.disabled = true;
            trainingStatus.innerHTML = '<div class="alert alert-info">Model eğitiliyor, lütfen bekleyin...</div>';
            
            const response = await fetch('/train', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                trainingStatus.innerHTML = '<div class="alert alert-success">Model başarıyla eğitildi!</div>';
                
                // Performans metriklerini göster
                modelPerformance.innerHTML = `
                    <h4>Model Performansı</h4>
                    <p><strong>ROC AUC Skoru:</strong> ${data.performans.roc_auc.toFixed(4)}</p>
                    <pre>${data.performans.classification_report}</pre>
                    <div class="mt-3">
                        <h5>AI Yorumu:</h5>
                        <p>${data.ai_interpretation.replace(/\n/g, '<br>')}</p>
                    </div>
                `;
            } else {
                trainingStatus.innerHTML = `<div class="alert alert-danger">Hata: ${data.message}</div>`;
            }
        } catch (error) {
            trainingStatus.innerHTML = `<div class="alert alert-danger">Hata: ${error.message}</div>`;
        } finally {
            trainButton.disabled = false;
        }
    });

    // Tahmin
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        try {
            const formData = {
                employed: document.getElementById('employed').value,
                bank_balance: document.getElementById('bankBalance').value,
                annual_salary: document.getElementById('annualSalary').value
            };
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                const resultHtml = `
                    <div class="alert ${data.prediction === 1 ? 'alert-warning' : 'alert-success'}">
                        <h4 class="alert-heading">Tahmin Sonucu</h4>
                        <p><strong>Temerrüt Tahmini:</strong> ${data.prediction === 1 ? 'Temerrüt Riski Var' : 'Temerrüt Riski Yok'}</p>
                        <p><strong>Olasılıklar:</strong></p>
                        <ul>
                            <li>Temerrüt Yok: %${(data.probability.no_default * 100).toFixed(2)}</li>
                            <li>Temerrüt Var: %${(data.probability.default * 100).toFixed(2)}</li>
                        </ul>
                        <hr>
                        <h5>Özellik Etkileri:</h5>
                        <ul>
                            ${data.explanation.map(exp => `
                                <li>${exp[0]}: ${exp[1] > 0 ? 'Pozitif' : 'Negatif'} etki (${Math.abs(exp[1]).toFixed(3)})</li>
                            `).join('')}
                        </ul>
                    </div>
                `;
                
                predictionResult.innerHTML = resultHtml;
            } else {
                predictionResult.innerHTML = `<div class="alert alert-danger">Hata: ${data.message}</div>`;
            }
        } catch (error) {
            predictionResult.innerHTML = `<div class="alert alert-danger">Hata: ${error.message}</div>`;
        }
    });
}); 