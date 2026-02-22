import React, { useState } from 'react';
import { Search, Database, Layers, CheckCircle2, Loader2, ArrowRight } from 'lucide-react';
import { apiClient } from '../api/client';
import type { Dataset } from '../api/client';

interface DatasetExplorerProps {
    sessionId: string;
    onDatasetsLoaded: () => void;
}

export function DatasetExplorer({ sessionId, onDatasetsLoaded }: DatasetExplorerProps) {
    const [query, setQuery] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [selectedDois, setSelectedDois] = useState<Set<string>>(new Set());
    const [isActivating, setIsActivating] = useState(false);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        setIsSearching(true);
        try {
            await apiClient.searchDatasets(sessionId, { query });
            const res = await apiClient.listDatasets(sessionId);
            if (res.datasets) {
                setDatasets(res.datasets);
            }
        } catch (err) {
            console.error("Search failed:", err);
        } finally {
            setIsSearching(false);
        }
    };

    const toggleSelection = (doi: string) => {
        const newSet = new Set(selectedDois);
        if (newSet.has(doi)) {
            newSet.delete(doi);
        } else {
            newSet.add(doi);
        }
        setSelectedDois(newSet);
    };

    const handleActivate = async () => {
        if (selectedDois.size === 0) return;
        setIsActivating(true);
        try {
            await apiClient.selectDatasets(sessionId, Array.from(selectedDois));
            onDatasetsLoaded();
        } catch (err) {
            console.error("Activation failed:", err);
        } finally {
            setIsActivating(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-transparent p-8 max-w-5xl mx-auto w-full relative z-10">
            <div className="mb-8">
                <h2 className="text-3xl font-display font-bold text-white flex items-center gap-3 mb-3 shrink-0">
                    <Database size={32} className="text-teal-400 drop-shadow-[0_0_8px_rgba(20,184,166,0.8)]" />
                    Dataset Explorer
                </h2>
                <p className="text-slate-400 text-lg">Search for Earth system datasets and select them for analysis.</p>
            </div>

            <form onSubmit={handleSearch} className="mb-8 relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-2xl blur opacity-25 group-focus-within:opacity-50 transition duration-500"></div>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search for datasets (e.g., 'sea surface temperature in the arctic')"
                    className="relative w-full glass-card text-white placeholder-slate-400 rounded-2xl pl-16 pr-36 py-5 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-transparent transition-all shadow-2xl text-lg backdrop-blur-xl"
                />
                <Search className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-teal-400 transition-colors" size={26} />
                <button
                    type="submit"
                    disabled={isSearching}
                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-gradient-to-r from-teal-500 to-teal-600 hover:from-teal-400 hover:to-teal-500 text-white px-8 py-2.5 rounded-xl font-semibold transition-all shadow-[0_0_15px_rgba(20,184,166,0.4)] disabled:opacity-50 flex items-center gap-2 hover:scale-105 active:scale-95"
                >
                    {isSearching ? <Loader2 size={20} className="animate-spin" /> : "Search"}
                </button>
            </form>

            {datasets.length > 0 && (
                <div className="flex justify-between items-center mb-6 glass-card p-4 rounded-xl border border-white/5">
                    <span className="font-semibold text-slate-300">
                        <span className="text-teal-400 font-bold">{datasets.length}</span> Results Found
                    </span>
                    <div className="flex items-center gap-5">
                        <span className="text-sm text-slate-400 font-medium bg-slate-900/50 px-3 py-1 rounded-lg border border-white/5">
                            <span className="text-teal-400">{selectedDois.size}</span> selected
                        </span>
                        <button
                            onClick={handleActivate}
                            disabled={selectedDois.size === 0 || isActivating}
                            className="bg-white hover:bg-slate-200 text-slate-900 px-6 py-2 rounded-lg font-bold transition-all disabled:opacity-50 flex items-center gap-2 shadow-[0_0_15px_rgba(255,255,255,0.2)] hover:shadow-[0_0_20px_rgba(255,255,255,0.4)] disabled:hover:shadow-none hover:-translate-y-0.5"
                        >
                            {isActivating ? <Loader2 size={18} className="animate-spin text-slate-600" /> : <ArrowRight size={18} />}
                            Load into Workspace
                        </button>
                    </div>
                </div>
            )}

            <div className="flex-1 overflow-y-auto pr-3 space-y-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                {datasets.map((ds, idx) => {
                    const doi = (ds.DOI || ds.doi) as string;
                    if (!doi) return null;
                    const isSelected = selectedDois.has(doi);
                    return (
                        <div
                            key={doi}
                            onClick={() => toggleSelection(doi)}
                            className={`p-6 rounded-2xl cursor-pointer transition-all duration-300 flex gap-5 backdrop-blur-md relative overflow-hidden group hover:-translate-y-1 ${isSelected
                                ? 'bg-teal-900/20 border border-teal-500/50 shadow-[0_0_20px_rgba(20,184,166,0.15)] ring-1 ring-teal-500/30'
                                : 'glass-card hover:bg-slate-800/60 hover:border-slate-600'
                                }`}
                            style={{ animationDelay: `${idx * 50}ms`, animationFillMode: 'both' }}
                        >
                            {isSelected && <div className="absolute inset-0 bg-gradient-to-r from-teal-500/10 to-transparent pointer-events-none"></div>}

                            <div className="pt-1 z-10">
                                <div className={`w-7 h-7 rounded-full border-2 flex items-center justify-center transition-all duration-300 shadow-inner ${isSelected
                                    ? 'border-teal-400 bg-teal-500 shadow-[0_0_10px_rgba(20,184,166,0.6)] scale-110'
                                    : 'border-slate-600 bg-slate-900/50 group-hover:border-slate-500'
                                    }`}>
                                    {isSelected && <CheckCircle2 size={18} className="text-white drop-shadow-md" />}
                                </div>
                            </div>
                            <div className="flex-1 z-10">
                                <div className="flex items-start justify-between mb-2">
                                    <h3 className="text-xl font-display font-semibold text-slate-100 leading-tight pr-4 group-hover:text-teal-300 transition-colors">
                                        {ds.Name || ds.name || ds.Title || ds.title}
                                    </h3>
                                    {(ds.Data_Type || ds.data_type) && (
                                        <span className="px-3 py-1.5 text-xs font-bold bg-slate-900/80 text-teal-400 border border-teal-500/20 rounded-lg whitespace-nowrap uppercase tracking-widest shadow-inner">
                                            {ds.Data_Type || ds.data_type}
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-5 text-sm text-slate-400 mb-4 font-medium">
                                    <span className="flex items-center gap-1.5 bg-slate-900/50 px-2 py-1 rounded-md border border-white/5">
                                        <Layers size={14} className="text-cyan-400" />
                                        {ds.Parameters ? ds.Parameters.split(',').length : (ds.params?.length || 0)} parameters
                                    </span>
                                    <span className="font-mono text-xs opacity-70">DOI: {doi}</span>
                                </div>
                                <p className="text-slate-400 text-sm line-clamp-2 leading-relaxed font-light">
                                    {ds.Description || ds.description || "No description available."}
                                </p>
                            </div>
                        </div>
                    );
                })}
                {datasets.length === 0 && !isSearching && (
                    <div className="h-full flex flex-col items-center justify-center text-slate-500 min-h-[400px]">
                        <div className="w-24 h-24 mb-6 rounded-full bg-slate-800/50 flex items-center justify-center border border-white/5 shadow-inner">
                            <Search size={40} className="text-slate-600" />
                        </div>
                        <p className="text-xl font-display font-medium text-slate-400">Enter a query to find relevant datasets</p>
                        <p className="text-sm mt-2 text-slate-500">Search Earth systems data easily.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
