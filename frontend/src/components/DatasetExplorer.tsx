import React, { useState } from 'react';
import { Search, Database, Layers, CheckCircle2, Loader2, ArrowRight, Bot, AlertCircle, ChevronUp, Table2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { apiClient } from '../api/client';
import type { Dataset } from '../api/client';

interface DatasetExplorerProps {
    sessionId: string;
    onDatasetsLoaded: () => void;
}

const EXAMPLE_QUERIES = [
    "Gelatinous zooplankton in the Fram Strait",
    "Continuous records of atmospheric greenhouse gases",
    "Renewable energy sources datasets",
    "Prokaryote abundance on Hakon Mosby volcano",
    "Sea surface temperature in the Arctic Ocean",
    "Physical oceanography CTD profiles MOSAiC",
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PreviewData = Record<string, any>;

export function DatasetExplorer({ sessionId, onDatasetsLoaded }: DatasetExplorerProps) {
    const [query, setQuery] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [selectedDois, setSelectedDois] = useState<Set<string>>(new Set());
    const [isActivating, setIsActivating] = useState(false);
    const [searchResponse, setSearchResponse] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [hasSearched, setHasSearched] = useState(false);
    const [searchMode] = useState<'simple' | 'deep'>('simple');
    const [expandedDoi, setExpandedDoi] = useState<string | null>(null);
    const [previewData, setPreviewData] = useState<PreviewData | null>(null);
    const [isLoadingPreview, setIsLoadingPreview] = useState(false);

    const handleSearch = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;

        setIsSearching(true);
        setErrorMessage(null);
        setSearchResponse(null);
        setExpandedDoi(null);
        setPreviewData(null);
        try {
            const searchResult = await apiClient.searchDatasets(sessionId, { query, search_mode: searchMode });
            if (searchResult.response) {
                setSearchResponse(searchResult.response);
            }
            const res = await apiClient.listDatasets(sessionId);
            if (res.datasets) {
                setDatasets(res.datasets);
            }
            setHasSearched(true);
        } catch (err) {
            console.error("Search failed:", err);
            setErrorMessage("Search failed. Please try again.");
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

    const handleSelectAll = () => {
        const allDois = datasets.map(ds => (ds.DOI || ds.doi) as string).filter(Boolean);
        if (selectedDois.size === allDois.length) {
            setSelectedDois(new Set());
        } else {
            setSelectedDois(new Set(allDois));
        }
    };

    const handleActivate = async () => {
        if (selectedDois.size === 0) return;
        setIsActivating(true);
        setErrorMessage(null);
        try {
            await apiClient.selectDatasets(sessionId, Array.from(selectedDois));
            onDatasetsLoaded();
        } catch (err) {
            console.error("Activation failed:", err);
            setErrorMessage("Failed to load datasets. The server returned an error — check backend logs.");
        } finally {
            setIsActivating(false);
        }
    };

    const handleExampleClick = (example: string) => {
        setQuery(example);
    };

    const handlePreviewToggle = async (doi: string) => {
        if (expandedDoi === doi) {
            setExpandedDoi(null);
            setPreviewData(null);
            return;
        }
        setExpandedDoi(doi);
        setPreviewData(null);
        setIsLoadingPreview(true);
        try {
            const data = await apiClient.previewDataset(sessionId, doi);
            setPreviewData(data);
        } catch {
            setPreviewData({ error: "Could not load preview. Dataset may not be cached yet — select and load it first." });
        } finally {
            setIsLoadingPreview(false);
        }
    };

    const allDois = datasets.map(ds => (ds.DOI || ds.doi) as string).filter(Boolean);
    const allSelected = allDois.length > 0 && selectedDois.size === allDois.length;

    return (
        <div className="flex flex-col h-full bg-transparent p-8 max-w-5xl mx-auto w-full relative z-10">
            <div className="mb-6">
                <h2 className="text-3xl font-display font-bold text-white flex items-center gap-3 mb-3 shrink-0">
                    <Database size={32} className="text-teal-400 drop-shadow-[0_0_8px_rgba(20,184,166,0.8)]" />
                    Dataset Explorer
                </h2>
                <p className="text-slate-400 text-lg">Search for Earth system datasets and select them for analysis.</p>
            </div>

            {/* Search bar */}
            <form onSubmit={handleSearch} className="mb-4 relative group">
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



            {/* Example query chips */}
            {!hasSearched && (
                <div className="mb-6 flex flex-wrap gap-2">
                    {EXAMPLE_QUERIES.map((eq) => (
                        <button
                            key={eq}
                            onClick={() => handleExampleClick(eq)}
                            className="px-4 py-2 text-sm rounded-full bg-slate-800/60 border border-white/10 text-slate-300 hover:text-teal-400 hover:border-teal-500/30 hover:bg-slate-800 transition-all cursor-pointer hover:shadow-[0_0_10px_rgba(20,184,166,0.15)]"
                        >
                            {eq}
                        </button>
                    ))}
                </div>
            )}

            {/* Error banner */}
            {errorMessage && (
                <div className="mb-4 flex items-center gap-3 text-sm font-medium text-red-300 bg-red-950/30 py-3 px-5 rounded-xl border border-red-500/30 shadow-inner">
                    <AlertCircle size={18} className="shrink-0 text-red-400" /> {errorMessage}
                </div>
            )}

            {/* Agent response bubble */}
            {searchResponse && (
                <div className="mb-6 glass-card p-5 rounded-2xl border border-white/5">
                    <div className="flex items-center gap-2 mb-3 text-sm font-display font-bold text-teal-400 tracking-wide uppercase">
                        <Bot size={16} /> Agent Response
                    </div>
                    <div className="prose prose-sm prose-invert max-w-none prose-p:text-slate-300 prose-p:leading-relaxed">
                        <ReactMarkdown>{searchResponse}</ReactMarkdown>
                    </div>
                </div>
            )}

            {/* Results bar */}
            {datasets.length > 0 && (
                <div className="flex justify-between items-center mb-4 glass-card p-4 rounded-xl border border-white/5">
                    <div className="flex items-center gap-4">
                        <span className="font-semibold text-slate-300">
                            <span className="text-teal-400 font-bold">{datasets.length}</span> Results
                        </span>
                        <button
                            onClick={handleSelectAll}
                            className="text-xs font-medium text-slate-400 hover:text-teal-400 border border-white/10 hover:border-teal-500/30 px-3 py-1.5 rounded-lg transition-all"
                        >
                            {allSelected ? "Deselect All" : "Select All"}
                        </button>
                    </div>
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

            {/* Dataset list */}
            <div className="flex-1 overflow-y-auto pr-3 space-y-3 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                {datasets.map((ds, idx) => {
                    const doi = (ds.DOI || ds.doi) as string;
                    if (!doi) return null;
                    const isSelected = selectedDois.has(doi);
                    const isExpanded = expandedDoi === doi;
                    return (
                        <div key={doi} style={{ animationDelay: `${idx * 50}ms`, animationFillMode: 'both' }}>
                            <div
                                className={`p-5 rounded-2xl cursor-pointer transition-all duration-300 flex gap-5 backdrop-blur-md relative overflow-hidden group hover:-translate-y-0.5 ${isSelected
                                    ? 'bg-teal-900/20 border border-teal-500/50 shadow-[0_0_20px_rgba(20,184,166,0.15)] ring-1 ring-teal-500/30'
                                    : 'glass-card hover:bg-slate-800/60 hover:border-slate-600'
                                    }`}
                            >
                                {isSelected && <div className="absolute inset-0 bg-gradient-to-r from-teal-500/10 to-transparent pointer-events-none"></div>}

                                <div className="pt-1 z-10" onClick={() => toggleSelection(doi)}>
                                    <div className={`w-7 h-7 rounded-full border-2 flex items-center justify-center transition-all duration-300 shadow-inner ${isSelected
                                        ? 'border-teal-400 bg-teal-500 shadow-[0_0_10px_rgba(20,184,166,0.6)] scale-110'
                                        : 'border-slate-600 bg-slate-900/50 group-hover:border-slate-500'
                                        }`}>
                                        {isSelected && <CheckCircle2 size={18} className="text-white drop-shadow-md" />}
                                    </div>
                                </div>
                                <div className="flex-1 z-10" onClick={() => toggleSelection(doi)}>
                                    <div className="flex items-start justify-between mb-2">
                                        <h3 className="text-lg font-display font-semibold text-slate-100 leading-tight pr-4 group-hover:text-teal-300 transition-colors">
                                            {ds.Name || ds.name || ds.Title || ds.title}
                                        </h3>
                                        {(ds.Data_Type || ds.data_type) && (
                                            <span className="px-3 py-1 text-xs font-bold bg-slate-900/80 text-teal-400 border border-teal-500/20 rounded-lg whitespace-nowrap uppercase tracking-widest shadow-inner">
                                                {ds.Data_Type || ds.data_type}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4 text-sm text-slate-400 mb-3 font-medium">
                                        <span className="flex items-center gap-1.5 bg-slate-900/50 px-2 py-1 rounded-md border border-white/5">
                                            <Layers size={14} className="text-cyan-400" />
                                            {ds.Parameters ? ds.Parameters.split(',').length : (ds.params?.length || 0)} params
                                        </span>
                                        <span className="font-mono text-xs opacity-70">DOI: {doi}</span>
                                    </div>
                                    <p className="text-slate-400 text-sm line-clamp-2 leading-relaxed font-light">
                                        {ds.Description || ds.description || "No description available."}
                                    </p>
                                </div>
                                {/* Preview toggle button */}
                                <div className="z-10 pt-1">
                                    <button
                                        onClick={(e) => { e.stopPropagation(); handlePreviewToggle(doi); }}
                                        className={`p-2 rounded-lg transition-all ${isExpanded
                                            ? 'text-teal-400 bg-teal-500/20 border border-teal-500/30'
                                            : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 border border-transparent'
                                            }`}
                                        title="Preview data"
                                    >
                                        {isExpanded ? <ChevronUp size={18} /> : <Table2 size={18} />}
                                    </button>
                                </div>
                            </div>

                            {/* Preview Panel */}
                            {isExpanded && (
                                <div className="mt-1 ml-12 mr-2 p-5 rounded-xl bg-slate-900/80 border border-white/10 shadow-inner animate-in fade-in slide-in-from-top-2 duration-300">
                                    {isLoadingPreview ? (
                                        <div className="flex items-center gap-3 text-slate-400 text-sm py-4 justify-center">
                                            <Loader2 size={18} className="animate-spin text-teal-400" /> Loading preview...
                                        </div>
                                    ) : previewData?.error ? (
                                        <div className="text-sm text-amber-400 flex items-center gap-2">
                                            <AlertCircle size={16} /> {previewData.error}
                                        </div>
                                    ) : previewData?.columns ? (
                                        <div>
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center gap-3 text-sm">
                                                    <span className="text-teal-400 font-mono font-bold">{previewData.preview_file || 'DataFrame'}</span>
                                                    <span className="text-slate-500">
                                                        {previewData.shape?.[0]} rows × {previewData.shape?.[1]} cols
                                                    </span>
                                                </div>
                                                <div className="text-xs text-slate-500 font-mono">
                                                    Columns: {previewData.columns.length}
                                                </div>
                                            </div>
                                            <div className="overflow-x-auto rounded-lg border border-white/5">
                                                <table className="w-full text-xs text-left">
                                                    <thead>
                                                        <tr className="bg-slate-800/80 border-b border-white/10">
                                                            {previewData.columns.slice(0, 8).map((col: string) => (
                                                                <th key={col} className="px-3 py-2 font-semibold text-teal-400 whitespace-nowrap">{col}</th>
                                                            ))}
                                                            {previewData.columns.length > 8 && (
                                                                <th className="px-3 py-2 text-slate-500">+{previewData.columns.length - 8} more</th>
                                                            )}
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                                        {previewData.head?.slice(0, 5).map((row: Record<string, any>, ri: number) => (
                                                            <tr key={ri} className="border-b border-white/5 hover:bg-slate-800/50">
                                                                {previewData.columns.slice(0, 8).map((col: string) => (
                                                                    <td key={col} className="px-3 py-1.5 text-slate-300 whitespace-nowrap max-w-[200px] truncate font-mono">
                                                                        {String(row[col] ?? '')}
                                                                    </td>
                                                                ))}
                                                                {previewData.columns.length > 8 && <td className="px-3 py-1.5 text-slate-500">…</td>}
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    ) : previewData?.files ? (
                                        <div className="text-sm text-slate-400">
                                            <p className="mb-2 text-slate-300 font-medium">Files in dataset:</p>
                                            <div className="flex flex-wrap gap-2">
                                                {previewData.files.map((f: string) => (
                                                    <span key={f} className="px-2 py-1 bg-slate-800 rounded text-xs font-mono border border-white/5">{f}</span>
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <p className="text-sm text-slate-500">No preview data available.</p>
                                    )}
                                </div>
                            )}
                        </div>
                    );
                })}
                {datasets.length === 0 && !isSearching && !hasSearched && (
                    <div className="h-full flex flex-col items-center justify-center text-slate-500 min-h-[300px]">
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
