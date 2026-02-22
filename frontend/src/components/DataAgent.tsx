import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Bot, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { AgentWebSocket } from '../api/client';

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
            () => setIsConnected(false)
        );

        // We could override the onopen if we extended AgentWebSocket to take it,
        // but for now we'll just set it directly after connect since we know it's a synchronous Object creation
        // A better pattern is to pass an onConnect callback to AgentWebSocket constructor.
        wsRef.current.connect();

        // Use a timeout to avoid synchronous setState during render cycle
        setTimeout(() => setIsConnected(true), 0);

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
        setActiveStatus("Initializing agents...");
    };

    return (
        <div className="flex flex-col h-full bg-transparent max-w-5xl mx-auto w-full relative z-10 border-x border-white/5 shadow-2xl">
            <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent" ref={scrollRef}>
                {messages.length === 0 && !activeStatus && (
                    <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-70">
                        <div className="w-20 h-20 mb-6 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center shadow-inner border border-white/5">
                            <Bot size={40} className="text-teal-500/50" />
                        </div>
                        <p className="text-xl font-display font-medium text-slate-400">Pangaea Data Agent</p>
                        <p className="text-sm mt-2">Ready to assist with data analysis.</p>
                    </div>
                )}

                {messages.map((m, idx) => (
                    <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-4 duration-500`}>
                        <div className={`max-w-[85%] rounded-3xl p-6 ${m.role === 'user'
                            ? 'bg-gradient-to-br from-teal-500 to-cyan-600 text-white shadow-[0_10px_25px_-5px_rgba(20,184,166,0.5)] rounded-tr-sm backdrop-blur-md'
                            : m.isError
                                ? 'bg-red-950/40 text-red-200 border border-red-500/30 rounded-tl-sm backdrop-blur-md shadow-lg shadow-red-900/20'
                                : 'glass-card text-slate-200 rounded-tl-sm shadow-[0_10px_30px_-10px_rgba(0,0,0,0.5)]'
                            }`}>
                            {m.role === 'assistant' && (
                                <div className="flex items-center gap-3 mb-4 text-sm font-display font-bold text-teal-400 tracking-wide uppercase">
                                    <div className="w-8 h-8 rounded-full bg-slate-900/80 flex items-center justify-center border border-teal-500/30 shadow-[0_0_10px_rgba(20,184,166,0.3)]">
                                        <Bot size={16} />
                                    </div>
                                    Pangaea Agent
                                </div>
                            )}

                            <div className={`prose prose-sm max-w-none ${m.role === 'user' ? 'prose-invert prose-p:text-white' : 'prose-invert prose-p:text-slate-300 prose-headings:text-slate-100 prose-a:text-teal-400'} prose-p:leading-relaxed prose-pre:bg-slate-950/80 prose-pre:border prose-pre:border-white/10 prose-pre:shadow-inner prose-pre:text-slate-300`}>
                                <ReactMarkdown>{m.content}</ReactMarkdown>
                            </div>

                            {m.plots && m.plots.length > 0 && (
                                <div className="mt-6 grid grid-cols-1 gap-6 bg-slate-950/50 p-6 rounded-2xl border border-white/5 shadow-inner">
                                    {m.plots.map((url, i) => (
                                        <img key={i} src={`http://localhost:8000${url}`} alt="Agent Plot" className="rounded-xl border border-white/10 mx-auto shadow-2xl max-h-[500px] object-contain bg-slate-900" />
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isProcessing && activeStatus && (
                    <div className="flex justify-start animate-in fade-in duration-300">
                        <div className="flex items-center gap-4 glass-card px-6 py-4 rounded-full text-sm font-medium text-slate-300 shadow-lg border-teal-500/20">
                            <div className="relative flex items-center justify-center">
                                <span className="absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-20 animate-ping"></span>
                                <Loader2 size={20} className="animate-spin text-teal-400 relative" />
                            </div>
                            <span className="bg-gradient-to-r from-slate-200 to-slate-400 bg-clip-text text-transparent">{activeStatus}</span>
                        </div>
                    </div>
                )}
            </div>

            <div className="p-6 bg-slate-900/60 backdrop-blur-2xl border-t border-white/10">
                {!isConnected && (
                    <div className="mb-4 flex items-center justify-center gap-3 text-sm font-medium text-amber-400 bg-amber-950/30 py-3 rounded-xl border border-amber-500/30 shadow-inner">
                        <AlertCircle size={18} className="animate-pulse" /> Disconnected from reasoning engine. Attempting to reconnect...
                    </div>
                )}
                <form onSubmit={handleSubmit} className="relative flex items-end shadow-[0_10px_40px_-10px_rgba(0,0,0,0.5)] rounded-3xl group glass-card border-white/10 focus-within:border-teal-500/50 focus-within:ring-2 focus-within:ring-teal-500/20 transition-all duration-300">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask the data agent to analyze the active datasets..."
                        className="w-full bg-transparent px-6 py-5 rounded-3xl focus:outline-none resize-none min-h-[64px] max-h-40 text-slate-200 placeholder-slate-500 text-lg leading-relaxed scrollbar-thin scrollbar-thumb-slate-700"
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
                        className="mb-3 mr-3 bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-400 hover:to-cyan-500 text-white p-3.5 rounded-2xl transition-all shadow-[0_0_15px_rgba(20,184,166,0.3)] disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none shrink-0 hover:scale-105 active:scale-95"
                    >
                        {isProcessing ? <Loader2 size={22} className="animate-spin" /> : <Send size={22} className="ml-1" />}
                    </button>
                </form>
            </div>
        </div>
    );
}
