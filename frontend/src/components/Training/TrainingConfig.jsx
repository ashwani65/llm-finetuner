import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Play, Square, Settings } from 'lucide-react';
import { trainingAPI, datasetAPI } from '../../services/api';

export default function TrainingConfig() {
  const queryClient = useQueryClient();
  const [config, setConfig] = useState({
    model_name: 'meta-llama/Llama-2-7b-hf',
    dataset_id: '',
    num_epochs: 3,
    batch_size: 4,
    learning_rate: 0.0002,
    lora_r: 16,
    lora_alpha: 32,
    quantization: '4bit',
  });

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: datasetAPI.list,
  });

  const { data: trainingJobs } = useQuery({
    queryKey: ['training-jobs'],
    queryFn: trainingAPI.list,
    refetchInterval: 5000,
  });

  const startTraining = useMutation({
    mutationFn: trainingAPI.start,
    onSuccess: () => {
      queryClient.invalidateQueries(['training-jobs']);
    },
  });

  const stopTraining = useMutation({
    mutationFn: trainingAPI.stop,
    onSuccess: () => {
      queryClient.invalidateQueries(['training-jobs']);
    },
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    startTraining.mutate(config);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Training Configuration</h1>
        <p className="text-gray-500 mt-1">
          Configure and start fine-tuning jobs
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Form */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Training Parameters
          </h3>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="label">Base Model</label>
              <select
                value={config.model_name}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                className="input"
              >
                <option value="meta-llama/Llama-2-7b-hf">Llama 2 7B</option>
                <option value="mistralai/Mistral-7B-v0.1">Mistral 7B</option>
              </select>
            </div>

            <div>
              <label className="label">Dataset</label>
              <select
                value={config.dataset_id}
                onChange={(e) => setConfig({ ...config, dataset_id: e.target.value })}
                className="input"
                required
              >
                <option value="">Select dataset...</option>
                {datasets?.data?.map((ds) => (
                  <option key={ds.id} value={ds.id}>{ds.name}</option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Epochs</label>
                <input
                  type="number"
                  value={config.num_epochs}
                  onChange={(e) => setConfig({ ...config, num_epochs: parseInt(e.target.value) })}
                  className="input"
                  min="1"
                />
              </div>
              <div>
                <label className="label">Batch Size</label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                  className="input"
                  min="1"
                />
              </div>
            </div>

            <div>
              <label className="label">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                value={config.learning_rate}
                onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                className="input"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">LoRA Rank</label>
                <input
                  type="number"
                  value={config.lora_r}
                  onChange={(e) => setConfig({ ...config, lora_r: parseInt(e.target.value) })}
                  className="input"
                />
              </div>
              <div>
                <label className="label">LoRA Alpha</label>
                <input
                  type="number"
                  value={config.lora_alpha}
                  onChange={(e) => setConfig({ ...config, lora_alpha: parseInt(e.target.value) })}
                  className="input"
                />
              </div>
            </div>

            <div>
              <label className="label">Quantization</label>
              <select
                value={config.quantization}
                onChange={(e) => setConfig({ ...config, quantization: e.target.value })}
                className="input"
              >
                <option value="none">None</option>
                <option value="4bit">4-bit (QLoRA)</option>
                <option value="8bit">8-bit</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={!config.dataset_id || startTraining.isPending}
              className="btn btn-primary w-full disabled:opacity-50"
            >
              <Play className="w-4 h-4 mr-2" />
              {startTraining.isPending ? 'Starting...' : 'Start Training'}
            </button>
          </form>
        </div>

        {/* Active Jobs */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Active Training Jobs</h3>
          <div className="space-y-3">
            {trainingJobs?.data?.filter(j => j.status === 'running').map((job) => (
              <div key={job.id} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium">{job.name}</h4>
                  <button
                    onClick={() => stopTraining.mutate(job.id)}
                    className="btn btn-danger btn-sm"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                </div>
                <div className="space-y-1 text-sm text-gray-600">
                  <p>Model: {job.model}</p>
                  <p>Epoch: {job.current_epoch} / {job.total_epochs}</p>
                  <p>Loss: {job.current_loss?.toFixed(4)}</p>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all"
                      style={{ width: `${(job.current_epoch / job.total_epochs) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )) || <p className="text-gray-500">No active training jobs</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
