import { useQuery } from '@tanstack/react-query';
import { Activity, Cpu, Database, Zap, TrendingUp, Clock } from 'lucide-react';
import { systemAPI, trainingAPI, modelAPI } from '../../services/api';
import StatCard from '../common/StatCard';
import TrainingChart from '../common/TrainingChart';

export default function Dashboard() {
  const { data: gpuStatus } = useQuery({
    queryKey: ['gpu-status'],
    queryFn: systemAPI.getGPUStatus,
    refetchInterval: 5000,
  });

  const { data: trainingJobs } = useQuery({
    queryKey: ['training-jobs'],
    queryFn: trainingAPI.list,
  });

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: modelAPI.list,
  });

  const stats = [
    {
      name: 'Active Training Jobs',
      value: trainingJobs?.data?.filter(j => j.status === 'running').length || 0,
      icon: Zap,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
    },
    {
      name: 'Total Models',
      value: models?.data?.length || 0,
      icon: Database,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
    },
    {
      name: 'GPU Utilization',
      value: `${gpuStatus?.data?.utilization || 0}%`,
      icon: Cpu,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
    },
    {
      name: 'Training Time',
      value: `${calculateTotalTrainingTime(trainingJobs?.data)}h`,
      icon: Clock,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Monitor your LLM fine-tuning pipeline
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <StatCard key={stat.name} {...stat} />
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Progress */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Training Progress</h3>
          <TrainingChart data={trainingJobs?.data} />
        </div>

        {/* Recent Activity */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Recent Activity
          </h3>
          <div className="space-y-3">
            {trainingJobs?.data?.slice(0, 5).map((job) => (
              <div key={job.id} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
                <div>
                  <p className="font-medium text-sm">{job.name}</p>
                  <p className="text-xs text-gray-500">{job.model}</p>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  job.status === 'running' ? 'bg-blue-100 text-blue-700' :
                  job.status === 'completed' ? 'bg-green-100 text-green-700' :
                  'bg-gray-100 text-gray-700'
                }`}>
                  {job.status}
                </span>
              </div>
            )) || (
              <p className="text-gray-500 text-sm">No recent activity</p>
            )}
          </div>
        </div>
      </div>

      {/* GPU Status */}
      {gpuStatus?.data && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">GPU Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500">GPU Name</p>
              <p className="text-lg font-semibold">{gpuStatus.data.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Memory Used</p>
              <p className="text-lg font-semibold">
                {gpuStatus.data.memory_used} / {gpuStatus.data.memory_total} GB
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Temperature</p>
              <p className="text-lg font-semibold">{gpuStatus.data.temperature}°C</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function calculateTotalTrainingTime(jobs) {
  if (!jobs) return 0;
  return jobs.reduce((acc, job) => acc + (job.duration || 0), 0).toFixed(1);
}
