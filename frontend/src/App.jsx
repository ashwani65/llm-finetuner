import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/common/Sidebar'
import Dashboard from './components/Dashboard/Dashboard'
import DatasetManager from './components/Dataset/DatasetManager'
import TrainingConfig from './components/Training/TrainingConfig'
import EvaluationView from './components/Evaluation/EvaluationView'
import DeploymentPanel from './components/Deployment/DeploymentPanel'

function App() {
  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dataset" element={<DatasetManager />} />
          <Route path="/training" element={<TrainingConfig />} />
          <Route path="/evaluation" element={<EvaluationView />} />
          <Route path="/deployment" element={<DeploymentPanel />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
