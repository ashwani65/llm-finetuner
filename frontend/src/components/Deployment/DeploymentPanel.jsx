import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Rocket, Send, Square, Activity } from 'lucide-react';
import { deploymentAPI, modelAPI } from '../../services/api';

export default function DeploymentPanel() {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState('');
  const [testPrompt, setTestPrompt] = useState('');
  const [testResult, setTestResult] = useState(null);

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: modelAPI.list,
  });

  const { data: deployments } = useQuery({
    queryKey: ['deployments'],
    queryFn: deploymentAPI.list,
    refetchInterval: 5000,
  });

  const deployMutation = useMutation({
    mutationFn: ({ modelId, config }) => deploymentAPI.deploy(modelId, config),
    onSuccess: () => {
      queryClient.invalidateQueries(['deployments']);
    },
  });

  const stopMutation = useMutation({
    mutationFn: deploymentAPI.stop,
    onSuccess: () => {
      queryClient.invalidateQueries(['deployments']);
    },
  });

  const testMutation = useMutation({
    mutationFn: ({ deploymentId, prompt }) =>
      deploymentAPI.test(deploymentId, prompt),
    onSuccess: (data) => {
      setTestResult(data.data);
    },
  });

  const handleDeploy = () => {
    if (!selectedModel) return;
    deployMutation.mutate({
      modelId: selectedModel,
      config: {
        port: 8000,
        tensor_parallel: 1,
      },
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Model Deployment</h1>
        <p className="text-gray-500 mt-1">
          Deploy models with vLLM for fast inference
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Deploy Section */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Rocket className="w-5 h-5" />
            Deploy Model
          </h3>
          <div className="space-y-4">
            <div>
              <label className="label">Select Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input"
              >
                <option value="">Choose a model...</option>
                {models?.data?.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={handleDeploy}
              disabled={!selectedModel || deployMutation.isPending}
              className="btn btn-primary w-full disabled:opacity-50"
            >
              {deployMutation.isPending ? 'Deploying...' : 'Deploy with vLLM'}
            </button>
          </div>
        </div>

        {/* Test Inference */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Send className="w-5 h-5" />
            Test Inference
          </h3>
          <div className="space-y-4">
            <div>
              <label className="label">Prompt</label>
              <textarea
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
                className="input"
                rows={3}
                placeholder="Enter your prompt..."
              />
            </div>
            <button
              onClick={() => {
                const activeDeployment = deployments?.data?.find(
                  (d) => d.status === 'running'
                );
                if (activeDeployment) {
                  testMutation.mutate({
                    deploymentId: activeDeployment.id,
                    prompt: testPrompt,
                  });
                }
              }}
              disabled={!testPrompt || testMutation.isPending}
              className="btn btn-primary w-full disabled:opacity-50"
            >
              {testMutation.isPending ? 'Generating...' : 'Test'}
            </button>

            {testResult && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-700 mb-2">Result:</p>
                <p className="text-sm text-gray-900 whitespace-pre-wrap">
                  {testResult.text}
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Generated {testResult.tokens_generated} tokens in {testResult.latency}ms
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Active Deployments */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Active Deployments
        </h3>
        <div className="space-y-3">
          {deployments?.data?.map((deployment) => (
            <div
              key={deployment.id}
              className="flex items-center justify-between p-4 border border-gray-200 rounded-lg"
            >
              <div className="flex-1">
                <h4 className="font-medium">{deployment.model_name}</h4>
                <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                  <span>Port: {deployment.port}</span>
                  <span>Requests: {deployment.request_count}</span>
                  <span>Avg Latency: {deployment.avg_latency}ms</span>
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    deployment.status === 'running'
                      ? 'bg-green-100 text-green-700'
                      : 'bg-gray-100 text-gray-700'
                  }`}>
                    {deployment.status}
                  </span>
                </div>
              </div>
              {deployment.status === 'running' && (
                <button
                  onClick={() => stopMutation.mutate(deployment.id)}
                  className="btn btn-danger"
                >
                  <Square className="w-4 h-4" />
                </button>
              )}
            </div>
          )) || <p className="text-gray-500">No active deployments</p>}
        </div>
      </div>
    </div>
  );
}
