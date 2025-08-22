import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const api = {
  getDashboardData: async () => {
    const response = await axiosInstance.get('/dashboard');
    return response.data;
  },

  getSegments: async (type: string) => {
    const response = await axiosInstance.get(`/segments/${type}`);
    return response.data;
  },

  getCohortAnalysis: async () => {
    const response = await axiosInstance.get('/cohort-analysis');
    return response.data;
  },

  calculateABTest: async (params: any) => {
    const response = await axiosInstance.post('/ab-test/calculate', params);
    return response.data;
  },

  getChurnAnalysis: async () => {
    const response = await axiosInstance.get('/churn-analysis');
    return response.data;
  },

  getRecommendations: async (customerId?: string) => {
    const url = customerId ? `/recommendations/${customerId}` : '/recommendations';
    const response = await axiosInstance.get(url);
    return response.data;
  },

  getBusinessInsights: async () => {
    const response = await axiosInstance.get('/business-insights');
    return response.data;
  },

  exportData: async (type: string, format: string = 'csv') => {
    const response = await axiosInstance.get(`/export/${type}`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },
};