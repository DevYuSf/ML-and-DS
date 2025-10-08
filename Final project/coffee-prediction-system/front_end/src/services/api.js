import axios from 'axios';

// Base URL for your Flask API
const API_BASE_URL = 'http://localhost:5000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service functions
export const coffeeAPI = {
  // Get prediction
  async getPrediction(inputs, selectedModel = null) {
    try {
      const payload = {
        hour: inputs.hour,
        weekday: inputs.weekday,
        month: inputs.month,
      };
      
      // Add model selection if provided
      if (selectedModel) {
        payload.model = selectedModel;
      }

      const response = await api.post('/predict', payload);
      return response.data;
    } catch (error) {
      console.error('Prediction API error:', error);
      throw new Error('Failed to get prediction. Make sure your Flask server is running on port 5000.');
    }
  },

  // Get batch predictions
  async getBatchPredictions(inputsArray) {
    try {
      const response = await api.post('/predict/batch', {
        inputs: inputsArray
      });
      return response.data;
    } catch (error) {
      console.error('Batch prediction API error:', error);
      throw error;
    }
  },

  // Get model information
  async getModelInfo() {
    try {
      const response = await api.get('/models');
      return response.data;
    } catch (error) {
      console.error('Model info API error:', error);
      throw error;
    }
  },

  // Change active model
  async selectModel(modelName) {
    try {
      const response = await api.post('/models/select', {
        model_name: modelName
      });
      return response.data;
    } catch (error) {
      console.error('Model selection API error:', error);
      throw error;
    }
  },

  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check API error:', error);
      throw error;
    }
  }
};

export default coffeeAPI;