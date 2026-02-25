import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Bot, AlertCircle, Trash2, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { AgentWebSocket } from '../api/client';
import { WorkspacePanel } from './WorkspacePanel';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    plots?: string[];
    status?: string;
    isError?: boolean;
}

export function DataAgent({ sessionId }: { sessionId: string }) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [activeStatus, setActiveStatus] = useState<string | null>(null);
    const wsRef = useRef<AgentWebSocket | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        wsRef.current = new AgentWebSocket(
            sessionId,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (msg: any) => {
                if (msg.type === 'status') {
                    setActiveStatus(msg.data.message);
                } else if (msg.type === 'response') {
                    setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: msg.data.content,
                        plots: msg.data.plot_urls
                    }]);
                    setActiveStatus(null);
                    setIsProcessing(false);
                } else if (msg.type === 'error') {
                    setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: msg.data.message,
                        isError: true
                    }]);
                    setActiveStatus(null);
                    setIsProcessing(false);
                }
            },
            () => {
                setIsConnected(false);
                setIsProcessing(false);
                setActiveStatus(null);
            },
            () => setIsConnected(true),
            () => setIsConnected(false)
        );

        wsRef.current.connect();

        return () => {
            wsRef.current?.disconnect();
        };
    }, [sessionId]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, activeStatus]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || !isConnected || isProcessing) return;

        setMessages(prev => [...prev, { role: 'user', content: input }]);
        wsRef.current?.sendQuery(input);
        setInput('');
        setIsProcessing(true);
        setActiveStatus("Initializing agents…");
    };

    const handleClearHistory = () => {
        setMessages([]);
    };

    const handleExportHistory = () => {
        const data = JSON.stringify(messages, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pangaea_session_${sessionId.substring(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="flex flex-col h-full bg-pg-bg max-w-5xl mx-auto w-full border-x border-pg-border">
            {/* Workspace — loaded datasets */}
            <WorkspacePanel sessionId={sessionId} />

            {/* Action bar */}
            {messages.length > 0 && (
                <div className="flex items-center justify-between px-5 py-2.5 border-b border-pg-border bg-pg-surface shrink-0">
                    <span className="text-xs text-txt-tertiary font-medium">
                        {messages.length} message{messages.length !== 1 ? 's' : ''}
                    </span>
                    <div className="flex items-center gap-1.5">
                        <button
                            onClick={handleExportHistory}
                            className="flex items-center gap-1 text-[11px] text-txt-tertiary hover:text-accent px-2.5 py-1 rounded-lg border border-pg-border hover:border-accent/30 transition-all"
                        >
                            <Download size={12} /> Export
                        </button>
                        <button
                            onClick={handleClearHistory}
                            className="flex items-center gap-1 text-[11px] text-txt-tertiary hover:text-status-danger px-2.5 py-1 rounded-lg border border-pg-border hover:border-status-danger/30 transition-all"
                        >
                            <Trash2 size={12} /> Clear
                        </button>
                    </div>
                </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-5 space-y-4 scrollbar-thin scrollbar-thumb-txt-tertiary/20 scrollbar-track-transparent" ref={scrollRef}>
                {messages.length === 0 && !activeStatus && (
                    <div className="h-full flex flex-col items-center justify-center text-txt-tertiary">
                        <div className="w-14 h-14 mb-4 rounded-2xl bg-pg-card flex items-center justify-center border border-pg-border">
                            <Bot size={28} className="text-txt-tertiary/50" />
                        </div>
                        <p className="text-lg font-display font-medium text-txt-secondary">Pangaea Data Agent</p>
                        <p className="text-xs mt-1 text-txt-tertiary">Ready to assist with data analysis.</p>
                    </div>
                )}

                {messages.map((m, idx) => (
                    <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
                        <div className={`max-w-[85%] rounded-2xl p-4 ${m.role === 'user'
                            ? 'bg-accent text-white rounded-tr-md'
                            : m.isError
                                ? 'bg-status-danger/8 text-status-danger border border-status-danger/20 rounded-tl-md'
                                : 'surface-card text-txt-primary rounded-tl-md'
                            }`}>
                            {m.role === 'assistant' && (
                                <div className="flex items-center gap-2 mb-3 text-xs font-display font-semibold text-accent tracking-wide uppercase">
                                    <div className="w-6 h-6 rounded-full bg-accent-subtle flex items-center justify-center border border-accent/20">
                                        <Bot size={13} />
                                    </div>
                                    Pangaea Agent
                                </div>
                            )}

                            <div className={`prose prose-sm max-w-none ${m.role === 'user'
                                ? 'prose-invert prose-p:text-white/90'
                                : 'prose-scientific prose-p:text-txt-secondary prose-headings:text-txt-primary prose-a:text-accent prose-strong:text-txt-primary'
                                } prose-p:leading-relaxed prose-pre:bg-pg-bg prose-pre:border prose-pre:border-pg-border prose-pre:text-txt-secondary text-sm`}>
                                <ReactMarkdown>{m.content}</ReactMarkdown>
                            </div>

                            {m.plots && m.plots.length > 0 && (
                                <div className="mt-4 grid grid-cols-1 gap-4 bg-pg-bg p-4 rounded-xl border border-pg-border">
                                    {m.plots.map((url, i) => (
                                        <img key={i} src={`http://127.0.0.1:8000${url}`} alt="Agent Plot" className="rounded-lg border border-pg-border mx-auto max-h-[500px] object-contain bg-pg-surface" />
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isProcessing && activeStatus && (
                    <div className="flex justify-start animate-fade-in">
                        <div className="flex items-center gap-3 surface-card px-4 py-3 rounded-full text-xs font-medium text-txt-secondary">
                            <Loader2 size={16} className="animate-spin text-accent" />
                            <span className="text-accent font-medium">{activeStatus}</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Input */}
            <div className="p-4 bg-pg-surface border-t border-pg-border">
                {!isConnected && (
                    <div className="mb-3 flex items-center justify-center gap-2 text-xs font-medium text-status-warning bg-status-warning/10 py-2.5 rounded-xl border border-status-warning/20">
                        <AlertCircle size={14} className="animate-pulse" /> Disconnected from reasoning engine. Reconnecting…
                    </div>
                )}
                <form onSubmit={handleSubmit} className="relative flex items-end rounded-2xl border border-pg-border bg-pg-card focus-within:border-accent/50 focus-within:ring-2 focus-within:ring-accent/15 transition-all duration-200">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask the data agent to analyze the active datasets…"
                        className="w-full bg-transparent px-4 py-3.5 rounded-2xl focus:outline-none resize-none min-h-[52px] max-h-36 text-txt-primary placeholder-txt-tertiary text-sm leading-relaxed scrollbar-thin scrollbar-thumb-txt-tertiary/20"
                        rows={1}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit(e);
                            }
                        }}
                    />
                    <button
                        type="submit"
                        disabled={!isConnected || isProcessing || !input.trim()}
                        className="mb-2 mr-2 bg-accent hover:bg-accent-hover text-white p-2.5 rounded-xl transition-all disabled:opacity-30 disabled:cursor-not-allowed shrink-0 hover:scale-105 active:scale-95"
                    >
                        {isProcessing ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
                    </button>
                </form>
            </div>
        </div>
    );
}
