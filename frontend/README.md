# LLM Fine-tuner Frontend

Modern web dashboard for managing LLM fine-tuning pipelines.

## Features

- **Dashboard**: Real-time monitoring of training jobs, GPU usage, and system metrics
- **Dataset Management**: Upload, validate, and manage training datasets
- **Training Configuration**: Configure and launch fine-tuning jobs with LoRA/QLoRA
- **Model Evaluation**: Evaluate models with BLEU, ROUGE, and custom metrics
- **Deployment**: Deploy models with vLLM and test inference

## Tech Stack

- React 18
- Vite
- Tailwind CSS
- React Query (TanStack Query)
- Recharts
- Lucide Icons
- Axios

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at http://localhost:3000

### Build for Production

```bash
npm run build
```

## Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

## API Integration

The frontend connects to the FastAPI backend running on port 8000. Make sure the backend is running before starting the frontend.

Start the backend:
```bash
cd ..
python -m src.serving.api
```

## Project Structure

```
frontend/
├── public/           # Static assets
├── src/
│   ├── components/   # React components
│   │   ├── Dashboard/
│   │   ├── Dataset/
│   │   ├── Training/
│   │   ├── Evaluation/
│   │   ├── Deployment/
│   │   └── common/
│   ├── services/     # API services
│   ├── hooks/        # Custom hooks
│   ├── utils/        # Utilities
│   ├── styles/       # CSS
│   ├── App.jsx       # Main app
│   └── main.jsx      # Entry point
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Screenshots

### Dashboard
Real-time monitoring of training jobs and system metrics

### Training Configuration
Easy-to-use interface for configuring fine-tuning parameters

### Model Evaluation
Comprehensive metrics and performance comparison

### Deployment
Deploy and test models with vLLM

## License

MIT
