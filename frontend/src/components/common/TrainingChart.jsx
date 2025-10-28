import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function TrainingChart({ data }) {
  // Transform data for chart
  const chartData = data?.map((job, index) => ({
    name: `Job ${index + 1}`,
    loss: job.final_loss || 0,
    accuracy: (job.accuracy || 0) * 100,
  })) || [];

  if (!chartData.length) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No training data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="loss" stroke="#3b82f6" name="Loss" />
        <Line type="monotone" dataKey="accuracy" stroke="#10b981" name="Accuracy (%)" />
      </LineChart>
    </ResponsiveContainer>
  );
}
