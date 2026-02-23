import { useState, useEffect } from 'react';
import { Bot, Map as MapIcon, Settings, Loader2, ChevronDown } from 'lucide-react';
import { apiClient } from './api/client';
import { DatasetExplorer } from './components/DatasetExplorer';
import { DataAgent } from './components/DataAgent';

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'explorer' | 'agent'>('explorer');
  const [isInitializing, setIsInitializing] = useState(true);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  // Initialize Session on Mount
  useEffect(() => {
    async function initSession() {
      try {
        // Fetch models list and create session in parallel
        const [session, modelsData] = await Promise.all([
          apiClient.createSession(),
          apiClient.getModels(),
        ]);
        setSessionId(session.session_id);
        setModels(modelsData.models);
        setSelectedModel(modelsData.default);
      } catch (err) {
        console.error("Failed to initialize session", err);
      } finally {
        setIsInitializing(false);
      }
    }
    initSession();
  }, []);

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    if (sessionId) {
      try {
        await apiClient.setModel(sessionId, newModel);
      } catch (err) {
        console.error("Failed to set model", err);
      }
    }
  };

  if (isInitializing || !sessionId) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-slate-950 text-slate-200">
        <Loader2 className="animate-spin text-teal-500 mb-4" size={48} />
        <h2 className="text-xl font-display font-semibold tracking-wide">Initializing Pangaea Engine...</h2>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans selection:bg-teal-500/30">
      {/* Sidebar - Navigation */}
      <aside className="w-16 bg-slate-900 border-r border-white/5 flex flex-col items-center py-6 gap-8 z-20 shadow-2xl">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-white font-display font-bold text-xl shadow-[0_0_15px_rgba(20,184,166,0.5)] ring-1 ring-white/20">
          P
        </div>
        <nav className="flex flex-col gap-4 flex-1">
          <button
            onClick={() => setActiveTab('explorer')}
            className={`p-3 rounded-xl transition-all duration-300 relative group ${activeTab === 'explorer'
              ? 'text-teal-400 bg-slate-800/80 shadow-inner'
              : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
          >
            {activeTab === 'explorer' && <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-teal-500 rounded-r-md shadow-[0_0_10px_rgba(20,184,166,0.8)]"></div>}
            <MapIcon size={24} className="group-hover:scale-110 transition-transform" />
          </button>
          <button
            onClick={() => setActiveTab('agent')}
            className={`p-3 rounded-xl transition-all duration-300 relative group ${activeTab === 'agent'
              ? 'text-teal-400 bg-slate-800/80 shadow-inner'
              : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
          >
            {activeTab === 'agent' && <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-teal-500 rounded-r-md shadow-[0_0_10px_rgba(20,184,166,0.8)]"></div>}
            <Bot size={24} className="group-hover:scale-110 transition-transform" />
          </button>
        </nav>
        <button className="p-3 rounded-xl text-slate-500 hover:text-slate-200 hover:bg-slate-800/50 transition-all duration-300 hover:rotate-45">
          <Settings size={24} />
        </button>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-slate-950">
        <header className="h-16 glass-panel border-b-0 border-r-0 border-l-0 flex items-center px-6 shrink-0 z-10 w-full">
          <h1 className="text-xl font-display font-semibold text-white tracking-tight flex items-center gap-2">
            Pangaea <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400 font-bold">GPT</span>
          </h1>
          <div className="ml-auto flex items-center gap-4">
            {/* Model Selector */}
            <div className="relative">
              <select
                id="model-selector"
                value={selectedModel}
                onChange={handleModelChange}
                className="appearance-none bg-slate-900/80 border border-white/10 text-slate-200 text-xs font-mono rounded-lg pl-3 pr-8 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-500/50 focus:border-teal-500/50 cursor-pointer hover:border-white/20 transition-colors shadow-inner"
              >
                {models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            </div>
            <span className="flex items-center gap-2 text-xs text-teal-400 bg-teal-400/10 px-3 py-1.5 rounded-full font-medium border border-teal-400/20 uppercase tracking-wider shadow-[0_0_10px_rgba(20,184,166,0.1)]">
              <span className="w-2 h-2 rounded-full bg-teal-400 shadow-[0_0_5px_rgba(20,184,166,0.8)] animate-pulse"></span>
              API Online
            </span>
            <span className="text-xs font-mono text-slate-400 bg-slate-900/80 border border-white/5 px-3 py-1.5 flex rounded-lg shadow-inner">
              Session: {sessionId.substring(0, 8)}
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-hidden relative">
          <div className={`absolute inset-0 transition-opacity duration-500 ${activeTab === 'explorer' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <DatasetExplorer
              sessionId={sessionId}
              onDatasetsLoaded={() => setActiveTab('agent')}
            />
          </div>
          <div className={`absolute inset-0 transition-opacity duration-500 flex justify-center ${activeTab === 'agent' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <DataAgent sessionId={sessionId} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

