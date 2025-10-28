import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Upload, Trash2, CheckCircle, AlertCircle, Database } from 'lucide-react';
import { datasetAPI } from '../../services/api';

export default function DatasetManager() {
  const [selectedFile, setSelectedFile] = useState(null);
  const queryClient = useQueryClient();

  const { data: datasets, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: datasetAPI.list,
  });

  const uploadMutation = useMutation({
    mutationFn: datasetAPI.upload,
    onSuccess: () => {
      queryClient.invalidateQueries(['datasets']);
      setSelectedFile(null);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: datasetAPI.delete,
    onSuccess: () => {
      queryClient.invalidateQueries(['datasets']);
    },
  });

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    uploadMutation.mutate(formData);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dataset Management</h1>
        <p className="text-gray-500 mt-1">
          Upload and manage training datasets
        </p>
      </div>

      {/* Upload Section */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Upload className="w-5 h-5" />
          Upload Dataset
        </h3>
        <div className="space-y-4">
          <div>
            <label className="label">Select JSON file</label>
            <input
              type="file"
              accept=".json"
              onChange={handleFileChange}
              className="input"
            />
          </div>
          {selectedFile && (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Database className="w-4 h-4" />
              {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
            </div>
          )}
          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploadMutation.isPending}
            className="btn btn-primary disabled:opacity-50"
          >
            {uploadMutation.isPending ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>

      {/* Datasets List */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Datasets</h3>
        {isLoading ? (
          <p className="text-gray-500">Loading...</p>
        ) : (
          <div className="space-y-3">
            {datasets?.data?.map((dataset) => (
              <div
                key={dataset.id}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg"
              >
                <div className="flex-1">
                  <h4 className="font-medium">{dataset.name}</h4>
                  <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                    <span>{dataset.examples} examples</span>
                    <span>{dataset.size}</span>
                    <span className="flex items-center gap-1">
                      {dataset.validated ? (
                        <>
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          Validated
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-4 h-4 text-yellow-600" />
                          Not validated
                        </>
                      )}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => deleteMutation.mutate(dataset.id)}
                  className="btn btn-danger"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            )) || <p className="text-gray-500">No datasets available</p>}
          </div>
        )}
      </div>
    </div>
  );
}
