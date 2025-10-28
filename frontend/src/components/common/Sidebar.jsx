import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  Zap,
  BarChart3,
  Rocket,
  Github,
} from 'lucide-react';

const navigation = [
  { name: 'Dashboard', to: '/', icon: LayoutDashboard },
  { name: 'Dataset', to: '/dataset', icon: Database },
  { name: 'Training', to: '/training', icon: Zap },
  { name: 'Evaluation', to: '/evaluation', icon: BarChart3 },
  { name: 'Deployment', to: '/deployment', icon: Rocket },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-200">
        <h1 className="text-2xl font-bold text-primary-600">LLM Fine-tuner</h1>
        <p className="text-sm text-gray-500 mt-1">Production Pipeline</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.to;

          return (
            <Link
              key={item.name}
              to={item.to}
              className={`
                flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors
                ${isActive
                  ? 'bg-primary-50 text-primary-700'
                  : 'text-gray-700 hover:bg-gray-50'
                }
              `}
            >
              <Icon className="w-5 h-5" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        <a
          href="https://github.com/ashwani65/llm-finetuner"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          <Github className="w-5 h-5" />
          View on GitHub
        </a>
      </div>
    </div>
  );
}
