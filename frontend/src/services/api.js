import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Dataset APIs
export const datasetAPI = {
  list: () => api.get('/datasets'),
  upload: (formData) => api.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  validate: (datasetId) => api.post(`/datasets/${datasetId}/validate`),
  delete: (datasetId) => api.delete(`/datasets/${datasetId}`),
  getStats: (datasetId) => api.get(`/datasets/${datasetId}/stats`),
};

// Training APIs
export const trainingAPI = {
  list: () => api.get('/training/jobs'),
  start: (config) => api.post('/training/start', config),
  stop: (jobId) => api.post(`/training/${jobId}/stop`),
  getStatus: (jobId) => api.get(`/training/${jobId}/status`),
  getMetrics: (jobId) => api.get(`/training/${jobId}/metrics`),
  getLogs: (jobId) => api.get(`/training/${jobId}/logs`),
};

// Evaluation APIs
export const evaluationAPI = {
  list: () => api.get('/evaluation/results'),
  evaluate: (modelId, testDataset) => api.post('/evaluation/evaluate', {
    model_id: modelId,
    test_dataset: testDataset,
  }),
  compare: (modelIds) => api.post('/evaluation/compare', { model_ids: modelIds }),
  getMetrics: (evaluationId) => api.get(`/evaluation/${evaluationId}/metrics`),
};

// Model APIs
export const modelAPI = {
  list: () => api.get('/models'),
  get: (modelId) => api.get(`/models/${modelId}`),
  delete: (modelId) => api.delete(`/models/${modelId}`),
  download: (modelId) => api.get(`/models/${modelId}/download`, {
    responseType: 'blob',
  }),
};

// Deployment APIs
export const deploymentAPI = {
  list: () => api.get('/deployments'),
  deploy: (modelId, config) => api.post('/deployments/deploy', {
    model_id: modelId,
    ...config,
  }),
  stop: (deploymentId) => api.post(`/deployments/${deploymentId}/stop`),
  getStatus: (deploymentId) => api.get(`/deployments/${deploymentId}/status`),
  test: (deploymentId, prompt) => api.post(`/deployments/${deploymentId}/test`, {
    prompt,
  }),
};

// MLflow APIs
export const mlflowAPI = {
  getExperiments: () => api.get('/mlflow/experiments'),
  getRuns: (experimentId) => api.get(`/mlflow/experiments/${experimentId}/runs`),
  getMetrics: (runId) => api.get(`/mlflow/runs/${runId}/metrics`),
};

// System APIs
export const systemAPI = {
  getGPUStatus: () => api.get('/system/gpu'),
  getSystemMetrics: () => api.get('/system/metrics'),
  health: () => api.get('/health'),
};

export default api;
