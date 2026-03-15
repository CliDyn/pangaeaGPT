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
            // Auto-select DOIs recommended by the agent
            if (searchResult.recommended_dois && searchResult.recommended_dois.length > 0) {
                setSelectedDois(new Set(searchResult.recommended_dois));
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
        setQuery(prev => prev.trim() ? `${prev.trim()} ${example}` : example);
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
        <div className="flex flex-col h-full bg-pg-bg p-6 lg:p-8 max-w-5xl mx-auto w-full">
            {/* Header */}
            <div className="mb-5">
                <h2 className="text-2xl font-display font-bold text-txt-primary flex items-center gap-2.5 mb-1.5">
                    <Database size={24} className="text-accent" />
                    Dataset Explorer
                </h2>
                <p className="text-txt-secondary text-sm">Search for Earth system datasets and select them for analysis.</p>
            </div>

            {/* Search bar */}
            <form onSubmit={handleSearch} className="mb-4 relative">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search for datasets (e.g., 'sea surface temperature in the arctic')"
                    className="w-full bg-pg-card text-txt-primary placeholder-txt-tertiary rounded-xl pl-12 pr-28 py-3.5 border border-pg-border focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent/50 transition-all shadow-input text-sm"
                />
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-txt-tertiary" size={18} />
                <button
                    type="submit"
                    disabled={isSearching}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-accent hover:bg-accent-hover text-white px-5 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-1.5"
                >
                    {isSearching ? <Loader2 size={16} className="animate-spin" /> : "Search"}
                </button>
            </form>

            {/* Example chips */}
            {!hasSearched && (
                <div className="mb-5 flex flex-wrap gap-1.5">
                    {EXAMPLE_QUERIES.map((eq) => (
                        <button
                            key={eq}
                            onClick={() => handleExampleClick(eq)}
                            className="px-3 py-1.5 text-xs rounded-full bg-pg-card border border-pg-border text-txt-secondary hover:text-accent hover:border-accent/30 hover:bg-accent-subtle transition-all cursor-pointer"
                        >
                            {eq}
                        </button>
                    ))}
                </div>
            )}

            {/* Scrollable content */}
            <div className="flex-1 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-txt-tertiary/20 scrollbar-track-transparent">
                {/* Error */}
                {errorMessage && (
                    <div className="mb-4 flex items-center gap-2.5 text-sm font-medium text-status-danger bg-status-danger/8 py-2.5 px-4 rounded-xl border border-status-danger/20">
                        <AlertCircle size={16} className="shrink-0" /> {errorMessage}
                    </div>
                )}

                {/* Agent response */}
                {searchResponse && (
                    <div className="mb-5 surface-card p-4 rounded-xl">
                        <div className="flex items-center gap-1.5 mb-2.5 text-xs font-display font-bold text-accent tracking-wide uppercase">
                            <Bot size={14} /> Agent Response
                        </div>
                        <div className="prose prose-sm prose-scientific max-w-none text-txt-secondary leading-relaxed">
                            <ReactMarkdown>{searchResponse}</ReactMarkdown>
                        </div>
                    </div>
                )}

                {/* Results bar */}
                {datasets.length > 0 && (
                    <div className="flex justify-between items-center mb-3 surface-card p-3 rounded-xl sticky top-0 z-20">
                        <div className="flex items-center gap-3">
                            <span className="text-sm text-txt-secondary">
                                <span className="text-accent font-semibold">{datasets.length}</span> Results
                            </span>
                            <button
                                onClick={handleSelectAll}
                                className="text-xs text-txt-tertiary hover:text-accent border border-pg-border hover:border-accent/30 px-2.5 py-1 rounded-lg transition-all"
                            >
                                {allSelected ? "Deselect All" : "Select All"}
                            </button>
                        </div>
                        <div className="flex items-center gap-3">
                            <span className="text-xs text-txt-tertiary font-medium bg-pg-card px-2.5 py-1 rounded-lg border border-pg-border">
                                <span className="text-accent">{selectedDois.size}</span> selected
                            </span>
                            <button
                                onClick={handleActivate}
                                disabled={selectedDois.size === 0 || isActivating}
                                className="bg-accent hover:bg-accent-hover text-white px-4 py-1.5 rounded-lg text-sm font-medium transition-all disabled:opacity-40 flex items-center gap-1.5 hover:-translate-y-px active:translate-y-0"
                            >
                                {isActivating ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
                                Load into Workspace
                            </button>
                        </div>
                    </div>
                )}

                {/* Dataset cards */}
                <div className="space-y-2">
                    {datasets.map((ds, idx) => {
                        const doi = (ds.DOI || ds.doi) as string;
                        if (!doi) return null;
                        const isSelected = selectedDois.has(doi);
                        const isExpanded = expandedDoi === doi;
                        return (
                            <div key={doi} className="animate-fade-in" style={{ animationDelay: `${idx * 30}ms`, animationFillMode: 'both' }}>
                                <div
                                    className={`p-4 rounded-xl cursor-pointer transition-all duration-200 flex gap-4 group hover:-translate-y-px ${isSelected
                                        ? 'bg-accent-subtle border border-accent/30 shadow-card'
                                        : 'surface-card hover:shadow-card-hover'
                                        }`}
                                >
                                    {/* Checkbox */}
                                    <div className="pt-0.5" onClick={() => toggleSelection(doi)}>
                                        <div className={`w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all duration-200 ${isSelected
                                            ? 'border-accent bg-accent'
                                            : 'border-pg-border group-hover:border-txt-tertiary'
                                            }`}>
                                            {isSelected && <CheckCircle2 size={14} className="text-white" />}
                                        </div>
                                    </div>

                                    {/* Content */}
                                    <div className="flex-1 min-w-0" onClick={() => toggleSelection(doi)}>
                                        <div className="flex items-start justify-between mb-1.5">
                                            <h3 className="text-sm font-semibold text-txt-primary leading-snug pr-3 group-hover:text-accent transition-colors">
                                                {ds.Name || ds.name || ds.Title || ds.title}
                                            </h3>
                                            {(ds.Data_Type || ds.data_type) && (
                                                <span className="px-2 py-0.5 text-[10px] font-medium bg-pg-card text-txt-tertiary border border-pg-border rounded-md whitespace-nowrap uppercase tracking-wider">
                                                    {ds.Data_Type || ds.data_type}
                                                </span>
                                            )}
                                        </div>
                                        <div className="flex items-center gap-3 text-xs text-txt-tertiary mb-2">
                                            <span className="flex items-center gap-1 bg-pg-card px-1.5 py-0.5 rounded border border-pg-border">
                                                <Layers size={11} className="text-accent" />
                                                {ds.Parameters ? ds.Parameters.split(',').length : (ds.params?.length || 0)} params
                                            </span>
                                            <span className="font-mono text-[10px]">DOI: {doi}</span>
                                        </div>
                                        <p className="text-txt-secondary text-xs line-clamp-2 leading-relaxed">
                                            {ds.Description || ds.description || "No description available."}
                                        </p>
                                    </div>

                                    {/* Preview toggle */}
                                    <div className="pt-0.5">
                                        <button
                                            onClick={(e) => { e.stopPropagation(); handlePreviewToggle(doi); }}
                                            className={`p-1.5 rounded-lg transition-all ${isExpanded
                                                ? 'text-accent bg-accent-subtle border border-accent/30'
                                                : 'text-txt-tertiary hover:text-accent hover:bg-accent-subtle border border-transparent'
                                                }`}
                                            title="Preview data"
                                        >
                                            {isExpanded ? <ChevronUp size={16} /> : <Table2 size={16} />}
                                        </button>
                                    </div>
                                </div>

                                {/* Preview Panel */}
                                {isExpanded && (
                                    <div className="mt-1 ml-9 mr-1 p-4 rounded-xl bg-pg-card border border-pg-border animate-slide-up">
                                        {isLoadingPreview ? (
                                            <div className="flex items-center gap-2 text-txt-tertiary text-xs py-3 justify-center">
                                                <Loader2 size={14} className="animate-spin text-accent" /> Loading preview…
                                            </div>
                                        ) : previewData?.error ? (
                                            <div className="text-xs text-status-warning flex items-center gap-1.5">
                                                <AlertCircle size={14} /> {previewData.error}
                                            </div>
                                        ) : previewData?.columns ? (
                                            <div>
                                                <div className="flex items-center justify-between mb-2.5">
                                                    <div className="flex items-center gap-2 text-xs">
                                                        <span className="text-accent font-mono font-medium">{previewData.preview_file || 'DataFrame'}</span>
                                                        <span className="text-txt-tertiary">
                                                            {previewData.shape?.[0]} rows × {previewData.shape?.[1]} cols
                                                        </span>
                                                    </div>
                                                    <span className="text-[10px] text-txt-tertiary font-mono">
                                                        Columns: {previewData.columns.length}
                                                    </span>
                                                </div>
                                                <div className="overflow-x-auto rounded-lg border border-pg-border">
                                                    <table className="w-full text-xs text-left">
                                                        <thead>
                                                            <tr className="bg-pg-surface border-b border-pg-border">
                                                                {previewData.columns.slice(0, 8).map((col: string) => (
                                                                    <th key={col} className="px-2.5 py-1.5 font-medium text-accent whitespace-nowrap">{col}</th>
                                                                ))}
                                                                {previewData.columns.length > 8 && (
                                                                    <th className="px-2.5 py-1.5 text-txt-tertiary">+{previewData.columns.length - 8} more</th>
                                                                )}
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                                            {previewData.head?.slice(0, 5).map((row: Record<string, any>, ri: number) => (
                                                                <tr key={ri} className="border-b border-pg-border/50 hover:bg-accent-subtle/50 transition-colors">
                                                                    {previewData.columns.slice(0, 8).map((col: string) => (
                                                                        <td key={col} className="px-2.5 py-1.5 text-txt-secondary whitespace-nowrap max-w-[180px] truncate font-mono text-[11px]">
                                                                            {String(row[col] ?? '')}
                                                                        </td>
                                                                    ))}
                                                                    {previewData.columns.length > 8 && <td className="px-2.5 py-1.5 text-txt-tertiary">…</td>}
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        ) : previewData?.files ? (
                                            <div className="text-xs text-txt-secondary">
                                                <p className="mb-1.5 text-txt-primary font-medium">Files in dataset:</p>
                                                <div className="flex flex-wrap gap-1.5">
                                                    {previewData.files.map((f: string) => (
                                                        <span key={f} className="px-2 py-0.5 bg-pg-surface rounded text-[11px] font-mono border border-pg-border">{f}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        ) : (
                                            <p className="text-xs text-txt-tertiary">No preview data available.</p>
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Empty state */}
                {datasets.length === 0 && !isSearching && !hasSearched && (
                    <div className="h-full flex flex-col items-center justify-center text-txt-tertiary min-h-[300px]">
                        <div className="w-16 h-16 mb-5 rounded-2xl bg-pg-card flex items-center justify-center border border-pg-border">
                            <Search size={28} className="text-txt-tertiary/50" />
                        </div>
                        <p className="text-lg font-display font-medium text-txt-secondary">Enter a query to find relevant datasets</p>
                        <p className="text-xs mt-1.5 text-txt-tertiary">Search Earth systems data easily.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
