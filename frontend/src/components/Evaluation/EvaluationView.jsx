import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { BarChart3, CheckCircle, TrendingUp } from 'lucide-react';
import { evaluationAPI, modelAPI } from '../../services/api';

export default function EvaluationView() {
  const [selectedModel, setSelectedModel] = useState('');

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: modelAPI.list,
  });

  const { data: evaluations } = useQuery({
    queryKey: ['evaluations'],
    queryFn: evaluationAPI.list,
  });

  const evaluateMutation = useMutation({
    mutationFn: ({ modelId, testDataset }) =>
      evaluationAPI.evaluate(modelId, testDataset),
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Model Evaluation</h1>
        <p className="text-gray-500 mt-1">
          Evaluate and compare model performance
        </p>
      </div>

      {/* Evaluate Section */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Run Evaluation
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
            onClick={() => evaluateMutation.mutate({ modelId: selectedModel })}
            disabled={!selectedModel || evaluateMutation.isPending}
            className="btn btn-primary disabled:opacity-50"
          >
            {evaluateMutation.isPending ? 'Evaluating...' : 'Run Evaluation'}
          </button>
        </div>
      </div>

      {/* Evaluation Results */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Recent Evaluations</h3>
        <div className="space-y-4">
          {evaluations?.data?.map((evaluation) => (
            <div key={evaluation.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="font-medium">{evaluation.model_name}</h4>
                  <p className="text-sm text-gray-500">{evaluation.timestamp}</p>
                </div>
                <CheckCircle className="w-5 h-5 text-green-600" />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  label="BLEU Score"
                  value={evaluation.metrics.bleu.toFixed(3)}
                  trend={evaluation.metrics.bleu_trend}
                />
                <MetricCard
                  label="Exact Match"
                  value={`${(evaluation.metrics.exact_match * 100).toFixed(1)}%`}
                  trend={evaluation.metrics.em_trend}
                />
                <MetricCard
                  label="ROUGE-L"
                  value={evaluation.metrics.rouge_l.toFixed(3)}
                  trend={evaluation.metrics.rouge_trend}
                />
                <MetricCard
                  label="Perplexity"
                  value={evaluation.metrics.perplexity.toFixed(2)}
                  trend={evaluation.metrics.perp_trend}
                  inverse
                />
              </div>
            </div>
          )) || <p className="text-gray-500">No evaluations available</p>}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, trend, inverse = false }) {
  const trendColor = trend > 0
    ? inverse ? 'text-red-600' : 'text-green-600'
    : inverse ? 'text-green-600' : 'text-red-600';

  return (
    <div className="bg-gray-50 p-3 rounded-lg">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
      {trend && (
        <div className={`flex items-center gap-1 text-xs ${trendColor} mt-1`}>
          <TrendingUp className="w-3 h-3" />
          {Math.abs(trend)}%
        </div>
      )}
    </div>
  );
}
