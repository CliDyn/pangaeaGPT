import { useState, useEffect, useCallback } from 'react';
import { ChevronRight, ChevronDown, Database, FileText, Table2, Folder, FolderOpen, Loader2, RefreshCw, X, ChevronLeft } from 'lucide-react';
import { apiClient } from '../api/client';

interface ActiveDataset {
    doi: string;
    name?: string;
    description?: string;
    data_type?: string;
    files?: string[];
    df_head?: Record<string, unknown>[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PreviewData = Record<string, any>;

export function WorkspacePanel({ sessionId }: { sessionId: string }) {
    const [datasets, setDatasets] = useState<ActiveDataset[]>([]);
    const [expandedDois, setExpandedDois] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [isPanelOpen, setIsPanelOpen] = useState(true);

    const [previewDoi, setPreviewDoi] = useState<string | null>(null);
    const [previewData, setPreviewData] = useState<PreviewData | null>(null);
    const [isLoadingPreview, setIsLoadingPreview] = useState(false);
    const [previewPage, setPreviewPage] = useState(0);
    const ROWS_PER_PAGE = 5;

    const fetchActiveDatasets = useCallback(async () => {
        setIsLoading(true);
        try {
            const data = await apiClient.getActiveDatasetsInfo(sessionId);
            if (data?.datasets && Array.isArray(data.datasets)) {
                setDatasets(data.datasets);
            } else {
                setDatasets([]);
            }
        } catch {
            // No datasets loaded yet
        } finally {
            setIsLoading(false);
        }
    }, [sessionId]);

    useEffect(() => {
        fetchActiveDatasets();
        const interval = setInterval(() => {
            fetchActiveDatasets();
        }, 5000);
        return () => clearInterval(interval);
    }, [fetchActiveDatasets]);


    const handleFileClick = async (doi: string, filename: string) => {
        const isTabular = /\.(csv|tab|tsv)$/i.test(filename);
        if (!isTabular) return;

        if (previewDoi === doi && previewData) {
            setPreviewDoi(null);
            setPreviewData(null);
            return;
        }

        setPreviewDoi(doi);
        setPreviewData(null);
        setIsLoadingPreview(true);
        setPreviewPage(0);
        try {
            const data = await apiClient.previewDataset(sessionId, doi);
            setPreviewData(data);
        } catch {
            setPreviewData({ error: "Could not load preview. Try selecting and loading the dataset first." });
        } finally {
            setIsLoadingPreview(false);
        }
    };

    const handleToggleExpand = async (doi: string) => {
        const next = new Set(expandedDois);
        const wasExpanded = next.has(doi);
        if (wasExpanded) {
            next.delete(doi);
            if (previewDoi === doi) {
                setPreviewDoi(null);
                setPreviewData(null);
            }
        } else {
            next.add(doi);
            if (!previewData || previewDoi !== doi) {
                setPreviewDoi(doi);
                setPreviewData(null);
                setIsLoadingPreview(true);
                setPreviewPage(0);
                try {
                    const data = await apiClient.previewDataset(sessionId, doi);
                    setPreviewData(data);
                } catch {
                    setPreviewData({ error: "Preview not available." });
                } finally {
                    setIsLoadingPreview(false);
                }
            }
        }
        setExpandedDois(next);
    };

    if (datasets.length === 0 && !isLoading) return null;

    const renderTablePreview = () => {
        if (!previewData) return null;
        if (previewData.error) {
            return <div className="text-xs text-status-warning px-3 py-2">{previewData.error}</div>;
        }
        if (!previewData.columns || !previewData.head) return null;

        const allRows: Record<string, unknown>[] = previewData.head;
        const totalPages = Math.ceil(allRows.length / ROWS_PER_PAGE);
        const pageRows = allRows.slice(previewPage * ROWS_PER_PAGE, (previewPage + 1) * ROWS_PER_PAGE);
        const cols: string[] = previewData.columns.slice(0, 10);
        const extraCols = previewData.columns.length > 10 ? previewData.columns.length - 10 : 0;

        return (
            <div className="mt-1.5 rounded-lg border border-pg-border bg-pg-bg overflow-hidden">
                {/* Preview header */}
                <div className="flex items-center justify-between px-2.5 py-1.5 bg-pg-surface border-b border-pg-border">
                    <div className="flex items-center gap-2 text-[11px]">
                        <span className="text-accent font-mono font-medium">{previewData.preview_file || 'DataFrame'}</span>
                        <span className="text-txt-tertiary">
                            {previewData.shape?.[0]} rows × {previewData.shape?.[1]} cols
                        </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        {totalPages > 1 && (
                            <div className="flex items-center gap-0.5 text-[10px] text-txt-tertiary">
                                <button
                                    onClick={() => setPreviewPage(p => Math.max(0, p - 1))}
                                    disabled={previewPage === 0}
                                    className="p-0.5 rounded hover:text-accent disabled:opacity-30 transition-colors"
                                >
                                    <ChevronLeft size={11} />
                                </button>
                                <span className="font-mono">{previewPage + 1}/{totalPages}</span>
                                <button
                                    onClick={() => setPreviewPage(p => Math.min(totalPages - 1, p + 1))}
                                    disabled={previewPage >= totalPages - 1}
                                    className="p-0.5 rounded hover:text-accent disabled:opacity-30 transition-colors"
                                >
                                    <ChevronRight size={11} />
                                </button>
                            </div>
                        )}
                        <button
                            onClick={() => { setPreviewDoi(null); setPreviewData(null); }}
                            className="p-0.5 text-txt-tertiary hover:text-status-danger transition-colors rounded"
                        >
                            <X size={11} />
                        </button>
                    </div>
                </div>
                {/* Table */}
                <div className="overflow-x-auto">
                    <table className="w-full text-[11px] text-left">
                        <thead>
                            <tr className="bg-pg-card border-b border-pg-border">
                                <th className="px-2 py-1 text-txt-tertiary font-mono w-7">#</th>
                                {cols.map((col) => (
                                    <th key={col} className="px-2 py-1 font-medium text-accent whitespace-nowrap">{col}</th>
                                ))}
                                {extraCols > 0 && (
                                    <th className="px-2 py-1 text-txt-tertiary text-[10px]">+{extraCols}</th>
                                )}
                            </tr>
                        </thead>
                        <tbody>
                            {pageRows.map((row, ri) => (
                                <tr key={ri} className="border-b border-pg-border/50 hover:bg-accent-subtle/50 transition-colors">
                                    <td className="px-2 py-1 text-txt-tertiary font-mono">{previewPage * ROWS_PER_PAGE + ri + 1}</td>
                                    {cols.map((col) => (
                                        <td key={col} className="px-2 py-1 text-txt-secondary whitespace-nowrap max-w-[150px] truncate font-mono">
                                            {String(row[col] ?? '')}
                                        </td>
                                    ))}
                                    {extraCols > 0 && <td className="px-2 py-1 text-txt-tertiary">…</td>}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        );
    };

    return (
        <div className="border-b border-pg-border bg-pg-surface shrink-0">
            {/* Panel header */}
            <div
                onClick={() => setIsPanelOpen(!isPanelOpen)}
                className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-pg-card transition-colors cursor-pointer select-none"
            >
                <div className="flex items-center gap-1.5 text-xs font-display font-semibold text-accent tracking-wide uppercase">
                    <Database size={13} />
                    Workspace
                    <span className="text-[10px] font-mono font-normal text-txt-tertiary ml-1">
                        {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}
                    </span>
                </div>
                <div className="flex items-center gap-1.5">
                    <button
                        onClick={(e) => { e.stopPropagation(); fetchActiveDatasets(); }}
                        className="p-1 text-txt-tertiary hover:text-accent transition-colors rounded"
                        title="Refresh"
                    >
                        <RefreshCw size={11} className={isLoading ? 'animate-spin' : ''} />
                    </button>
                    {isPanelOpen ? <ChevronDown size={12} className="text-txt-tertiary" /> : <ChevronRight size={12} className="text-txt-tertiary" />}
                </div>
            </div>

            {/* Panel content */}
            {isPanelOpen && (
                <div className="px-3 pb-2.5 max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-thumb-txt-tertiary/20 scrollbar-track-transparent">
                    {isLoading && datasets.length === 0 ? (
                        <div className="flex items-center justify-center gap-2 py-3 text-xs text-txt-tertiary">
                            <Loader2 size={13} className="animate-spin text-accent" /> Loading workspace…
                        </div>
                    ) : (
                        <div className="space-y-0.5">
                            {datasets.map((ds) => {
                                const isExpanded = expandedDois.has(ds.doi);
                                return (
                                    <div key={ds.doi}>
                                        <div
                                            onClick={() => handleToggleExpand(ds.doi)}
                                            className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded-lg hover:bg-pg-card transition-colors text-left group cursor-pointer"
                                        >
                                            {isExpanded
                                                ? <FolderOpen size={13} className="text-accent shrink-0" />
                                                : <Folder size={13} className="text-txt-tertiary group-hover:text-accent shrink-0 transition-colors" />
                                            }
                                            <span className="text-xs text-txt-primary truncate flex-1 font-medium">
                                                {ds.name || ds.doi}
                                            </span>
                                            {ds.data_type && (
                                                <span className="text-[9px] font-mono text-txt-tertiary uppercase tracking-wider shrink-0">
                                                    {ds.data_type}
                                                </span>
                                            )}
                                            {isExpanded
                                                ? <ChevronDown size={11} className="text-txt-tertiary shrink-0" />
                                                : <ChevronRight size={11} className="text-txt-tertiary shrink-0" />
                                            }
                                        </div>

                                        {isExpanded && (
                                            <div className="ml-4 pl-2.5 border-l border-pg-border mt-0.5 mb-1.5 space-y-0.5">
                                                <div className="text-[10px] font-mono text-txt-tertiary px-2 py-0.5 truncate">
                                                    {ds.doi}
                                                </div>

                                                {ds.description && (
                                                    <p className="text-[11px] text-txt-tertiary px-2 py-0.5 leading-relaxed line-clamp-2">
                                                        {ds.description}
                                                    </p>
                                                )}

                                                {ds.files && ds.files.length > 0 && (
                                                    <div className="space-y-0.5">
                                                        {ds.files.map((f) => {
                                                            const isTabular = /\.(csv|tab|tsv)$/i.test(f);
                                                            const isActive = previewDoi === ds.doi && previewData && !previewData.error;
                                                            return (
                                                                <div
                                                                    key={f}
                                                                    onClick={(e) => { e.stopPropagation(); if (isTabular) handleFileClick(ds.doi, f); }}
                                                                    className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${isTabular
                                                                        ? 'cursor-pointer hover:bg-accent-subtle group/file'
                                                                        : ''
                                                                        } ${isActive && previewDoi === ds.doi ? 'bg-accent-subtle' : ''}`}
                                                                >
                                                                    {isTabular
                                                                        ? <Table2 size={11} className={`shrink-0 transition-colors ${isActive && previewDoi === ds.doi ? 'text-accent' : 'text-status-success group-hover/file:text-accent'}`} />
                                                                        : <FileText size={11} className="text-txt-tertiary shrink-0" />
                                                                    }
                                                                    <span className={`text-[11px] font-mono truncate ${isTabular ? 'text-txt-secondary group-hover/file:text-accent transition-colors' : 'text-txt-tertiary'}`}>
                                                                        {f}
                                                                    </span>
                                                                    {isTabular && (
                                                                        <span className="text-[9px] text-accent/40 ml-auto shrink-0 uppercase tracking-wider opacity-0 group-hover/file:opacity-100 transition-opacity">
                                                                            preview
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                )}

                                                {previewDoi === ds.doi && (
                                                    <>
                                                        {isLoadingPreview ? (
                                                            <div className="flex items-center gap-1.5 py-2 px-2 text-[11px] text-txt-tertiary">
                                                                <Loader2 size={11} className="animate-spin text-accent" /> Loading preview…
                                                            </div>
                                                        ) : (
                                                            renderTablePreview()
                                                        )}
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
