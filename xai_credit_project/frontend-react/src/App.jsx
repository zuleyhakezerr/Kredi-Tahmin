import React, { useState } from 'react';
import { Container, Box, Button, TextField, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import axios from 'axios';

function App() {
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [formData, setFormData] = useState({
    employed: '1',
    bankBalance: '',
    annualSalary: ''
  });

  const handleTrainModel = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/train');
      setModelStatus(response.data);
    } catch (error) {
      setModelStatus({ success: false, message: error.message });
    }
    setLoading(false);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post('/api/predict', {
        employed: formData.employed,
        bank_balance: formData.bankBalance,
        annual_salary: formData.annualSalary
      });
      setPrediction(response.data);
    } catch (error) {
      setPrediction({ success: false, message: error.message });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Kredi Temerrüt Tahmin Sistemi
        </Typography>

        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Model Eğitimi
          </Typography>
          <Button
            variant="contained"
            onClick={handleTrainModel}
            disabled={loading}
            fullWidth
          >
            {loading ? <CircularProgress size={24} /> : 'Modeli Eğit'}
          </Button>
          {modelStatus && (
            <Box sx={{ mt: 2 }}>
              <Alert severity={modelStatus.success ? 'success' : 'error'}>
                {modelStatus.message}
              </Alert>
              {modelStatus.success && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1">Model Performansı:</Typography>
                  <pre>{JSON.stringify(modelStatus.performans, null, 2)}</pre>
                  <Typography variant="subtitle1">AI Yorumu:</Typography>
                  <Typography>{modelStatus.ai_interpretation}</Typography>
                </Box>
              )}
            </Box>
          )}
        </Paper>

        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Tahmin
          </Typography>
          <form onSubmit={handlePredict}>
            <Box sx={{ mb: 2 }}>
              <TextField
                select
                label="İstihdam Durumu"
                value={formData.employed}
                onChange={(e) => setFormData({ ...formData, employed: e.target.value })}
                fullWidth
                SelectProps={{
                  native: true
                }}
              >
                <option value="1">Çalışıyor</option>
                <option value="0">Çalışmıyor</option>
              </TextField>
            </Box>
            <Box sx={{ mb: 2 }}>
              <TextField
                type="number"
                label="Banka Bakiyesi"
                value={formData.bankBalance}
                onChange={(e) => setFormData({ ...formData, bankBalance: e.target.value })}
                fullWidth
                required
              />
            </Box>
            <Box sx={{ mb: 2 }}>
              <TextField
                type="number"
                label="Yıllık Maaş"
                value={formData.annualSalary}
                onChange={(e) => setFormData({ ...formData, annualSalary: e.target.value })}
                fullWidth
                required
              />
            </Box>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Tahmin Et'}
            </Button>
          </form>

          {prediction && (
            <Box sx={{ mt: 2 }}>
              {prediction.success ? (
                <Box>
                  <Alert severity={prediction.prediction === 1 ? 'warning' : 'success'}>
                    <Typography variant="h6">
                      {prediction.prediction === 1 ? 'Temerrüt Riski Var' : 'Temerrüt Riski Yok'}
                    </Typography>
                    <Typography>
                      Temerrüt Olasılığı: %{(prediction.probability.default * 100).toFixed(2)}
                    </Typography>
                  </Alert>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle1">Özellik Etkileri:</Typography>
                    {prediction.explanation.map((exp, index) => (
                      <Typography key={index}>
                        {exp[0]}: {exp[1] > 0 ? 'Pozitif' : 'Negatif'} etki ({Math.abs(exp[1]).toFixed(3)})
                      </Typography>
                    ))}
                  </Box>
                </Box>
              ) : (
                <Alert severity="error">{prediction.message}</Alert>
              )}
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default App; 