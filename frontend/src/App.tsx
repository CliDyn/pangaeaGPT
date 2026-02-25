import { useState, useEffect } from 'react';
import { Bot, Map as MapIcon, Settings, Loader2, ChevronDown, AlertCircle, Sun, Moon } from 'lucide-react';
import { apiClient } from './api/client';
import { DatasetExplorer } from './components/DatasetExplorer';
import { DataAgent } from './components/DataAgent';

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'explorer' | 'agent'>('explorer');
  const [isInitializing, setIsInitializing] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('pangaea-theme') as 'dark' | 'light') || 'dark';
  });

  const toggleTheme = () => {
    setTheme(prev => {
      const next = prev === 'dark' ? 'light' : 'dark';
      localStorage.setItem('pangaea-theme', next);
      return next;
    });
  };

  const initSession = async () => {
    setIsInitializing(true);
    setInitError(null);
    try {
      const [session, modelsData] = await Promise.all([
        apiClient.createSession(),
        apiClient.getModels(),
      ]);
      setSessionId(session.session_id);
      setModels(modelsData.models);
      setSelectedModel(modelsData.default);
    } catch (err) {
      console.error("Failed to initialize session", err);
      setInitError("Failed to connect to the backend server. Please ensure the Python backend is running.");
    } finally {
      setIsInitializing(false);
    }
  };

  useEffect(() => {
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
      <div className={`${theme === 'light' ? 'light' : ''}`}>
        <div className="flex flex-col items-center justify-center h-screen bg-pg-bg text-txt-primary transition-colors duration-300">
          {initError ? (
            <div className="w-14 h-14 rounded-2xl bg-status-danger/10 flex items-center justify-center text-status-danger border border-status-danger/20">
              <AlertCircle size={28} />
            </div>
          ) : (
            <Loader2 className="animate-spin text-accent" size={40} />
          )}
          <h2 className={`text-lg font-display font-semibold tracking-wide mt-5 ${initError ? 'text-status-danger' : 'text-txt-primary'}`}>
            {initError ? "Connection Error" : "Initializing Pangaea Engine…"}
          </h2>
          {initError && (
            <div className="mt-5 flex flex-col items-center max-w-md text-center">
              <p className="text-txt-secondary text-sm mb-5">{initError}</p>
              <button
                onClick={initSession}
                className="bg-pg-card hover:bg-accent-subtle border border-pg-border text-txt-primary px-5 py-2 rounded-xl transition-all flex items-center gap-2 text-sm font-medium hover:border-accent/40"
              >
                <Loader2 size={16} className="text-accent" /> Try Again
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`${theme === 'light' ? 'light' : ''}`}>
      <div className="flex h-screen bg-pg-bg text-txt-primary font-sans transition-colors duration-300">
        {/* Sidebar */}
        <aside className="w-14 bg-pg-surface border-r border-pg-border flex flex-col items-center py-5 gap-6 z-20 shrink-0">
          {/* Logo */}
          <div className="w-9 h-9 rounded-xl bg-accent flex items-center justify-center text-white font-display font-bold text-base select-none">
            P
          </div>

          <nav className="flex flex-col gap-2 flex-1">
            <button
              onClick={() => setActiveTab('explorer')}
              className={`p-2.5 rounded-xl transition-all duration-200 relative group ${activeTab === 'explorer'
                ? 'text-accent bg-accent-subtle'
                : 'text-txt-tertiary hover:text-txt-secondary hover:bg-pg-card'
                }`}
              title="Dataset Explorer"
            >
              {activeTab === 'explorer' && <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 bg-accent rounded-r-full"></div>}
              <MapIcon size={20} />
            </button>
            <button
              onClick={() => setActiveTab('agent')}
              className={`p-2.5 rounded-xl transition-all duration-200 relative group ${activeTab === 'agent'
                ? 'text-accent bg-accent-subtle'
                : 'text-txt-tertiary hover:text-txt-secondary hover:bg-pg-card'
                }`}
              title="Data Agent"
            >
              {activeTab === 'agent' && <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 bg-accent rounded-r-full"></div>}
              <Bot size={20} />
            </button>
          </nav>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2.5 rounded-xl text-txt-tertiary hover:text-txt-secondary hover:bg-pg-card transition-all duration-200"
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <button className="p-2.5 rounded-xl text-txt-tertiary hover:text-txt-secondary hover:bg-pg-card transition-all duration-200" title="Settings">
            <Settings size={18} />
          </button>
        </aside>

        {/* Main */}
        <main className="flex-1 flex flex-col h-full overflow-hidden">
          {/* Header */}
          <header className="h-14 bg-pg-surface border-b border-pg-border flex items-center px-5 shrink-0 z-10">
            <h1 className="text-base font-display font-semibold text-txt-primary tracking-tight flex items-center gap-1.5">
              Pangaea<span className="text-accent font-bold">GPT</span>
            </h1>
            <div className="ml-auto flex items-center gap-3">
              {/* Model Selector */}
              <div className="relative">
                <select
                  id="model-selector"
                  value={selectedModel}
                  onChange={handleModelChange}
                  className="appearance-none bg-pg-card border border-pg-border text-txt-primary text-xs font-mono rounded-lg pl-3 pr-7 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 cursor-pointer hover:border-txt-tertiary transition-colors"
                >
                  {models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
                <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-txt-tertiary pointer-events-none" />
              </div>
              {/* Status */}
              <span className="flex items-center gap-1.5 text-[11px] text-status-success bg-status-success/10 px-2.5 py-1 rounded-full font-medium border border-status-success/20 uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-status-success"></span>
                Online
              </span>
              {/* Session */}
              <span className="text-[11px] font-mono text-txt-tertiary bg-pg-card border border-pg-border px-2.5 py-1 rounded-lg">
                {sessionId.substring(0, 8)}
              </span>
            </div>
          </header>

          <div className="flex-1 overflow-hidden relative">
            <div className={`absolute inset-0 transition-opacity duration-300 ${activeTab === 'explorer' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
              <DatasetExplorer
                sessionId={sessionId}
                onDatasetsLoaded={() => setActiveTab('agent')}
              />
            </div>
            <div className={`absolute inset-0 transition-opacity duration-300 flex justify-center ${activeTab === 'agent' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
              <DataAgent sessionId={sessionId} />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
