import axios from 'axios';

const API_BASE_URL = '/api';
const WS_BASE_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api`;

// --- Types ---
export interface Dataset {
    doi?: string;
    DOI?: string;
    name?: string;
    Name?: string;
    title?: string;
    Title?: string;
    description?: string;
    Description?: string;
    data_type?: string;
    Data_Type?: string;
    params?: unknown[];
    Parameters?: string;
}

export interface SessionResponse {
    session_id: string;
    thread_id: string;
    search_message_count?: number;
    agent_message_count?: number;
    active_datasets?: Dataset[];
    has_datasets?: boolean;
}

export interface SearchRequest {
    query: string;
    search_mode?: string;
}

// --- REST API Client ---
export const apiClient = {
    // Session Management
    createSession: async (): Promise<SessionResponse> => {
        const res = await axios.post(`${API_BASE_URL}/sessions`);
        return res.data;
    },

    getSession: async (sessionId: string): Promise<SessionResponse> => {
        const res = await axios.get(`${API_BASE_URL}/sessions/${sessionId}`);
        return res.data;
    },

    // Dataset Search and Selection
    searchDatasets: async (sessionId: string, params: SearchRequest) => {
        const res = await axios.post(`${API_BASE_URL}/sessions/${sessionId}/search`, params);
        return res.data;
    },

    listDatasets: async (sessionId: string) => {
        const res = await axios.get(`${API_BASE_URL}/sessions/${sessionId}/datasets`);
        return res.data;
    },

    selectDatasets: async (sessionId: string, dois: string[]) => {
        const res = await axios.post(`${API_BASE_URL}/sessions/${sessionId}/datasets/select`, { dois });
        return res.data;
    },

    getActiveDatasetsInfo: async (sessionId: string) => {
        const res = await axios.get(`${API_BASE_URL}/sessions/${sessionId}/datasets/info`);
        return res.data;
    },

    // Model Selection
    getModels: async (): Promise<{ models: string[]; default: string }> => {
        const res = await axios.get(`${API_BASE_URL}/models`);
        return res.data;
    },

    setModel: async (sessionId: string, modelName: string) => {
        const res = await axios.put(`${API_BASE_URL}/sessions/${sessionId}/model`, { model_name: modelName });
        return res.data;
    },

    // Dataset Preview
    previewDataset: async (sessionId: string, doi: string) => {
        const res = await axios.get(`${API_BASE_URL}/sessions/${sessionId}/datasets/preview`, { params: { doi } });
        return res.data;
    },
};

// --- WebSocket Service ---
type WSMessageCallback = (msg: unknown) => void;

export class AgentWebSocket {
    private ws: WebSocket | null = null;
    private onMessage: WSMessageCallback;
    private onError: WSMessageCallback;
    private onOpen: () => void;
    private onClose: () => void;
    private url: string;

    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 5;
    private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
    private isIntentionalDisconnect: boolean = false;

    constructor(
        sessionId: string,
        onMessage: WSMessageCallback,
        onError: WSMessageCallback,
        onOpen: () => void,
        onClose: () => void
    ) {
        this.url = `${WS_BASE_URL}/sessions/${sessionId}/agent/ws`;
        this.onMessage = onMessage;
        this.onError = onError;
        this.onOpen = onOpen;
        this.onClose = onClose;
    }

    connect() {
        if (this.ws?.readyState === WebSocket.CONNECTING || this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        this.isIntentionalDisconnect = false;
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log("WebSocket connection established");
            this.reconnectAttempts = 0; // Reset attempts on successful connection
            this.onOpen();
        };
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.onMessage(data);
            } catch (err) {
                console.error("Failed to parse WS message", err);
            }
        };

        this.ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            this.onError({ type: "error", data: { message: "WebSocket connection error" } });
        };

        this.ws.onclose = () => {
            console.log("WebSocket connection closed");
            this.ws = null;
            this.onClose();
            this.attemptReconnect();
        };
    }

    private attemptReconnect() {
        if (this.isIntentionalDisconnect) return;

        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            // Exponential backoff: 1s, 2s, 4s, 8s, 16s... max 30s
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);

            console.log(`Attempting to reconnect in ${delay / 1000}s (Attempt ${this.reconnectAttempts})`);

            if (this.reconnectTimeoutId) clearTimeout(this.reconnectTimeoutId);

            this.reconnectTimeoutId = setTimeout(() => {
                console.log("Reconnecting WebSocket...");
                this.connect();
            }, delay);
        } else {
            console.error("Max WebSocket reconnect attempts reached. Please refresh the page.");
            this.onError({ type: "error", data: { message: "Lost connection to agent server. Please refresh." } });
        }
    }

    sendQuery(query: string) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ query }));
        } else {
            console.error("WebSocket is not connected");
            this.onError({ type: "error", data: { message: "Not connected to agent server" } });
        }
    }

    disconnect() {
        this.isIntentionalDisconnect = true;
        if (this.reconnectTimeoutId) {
            clearTimeout(this.reconnectTimeoutId);
            this.reconnectTimeoutId = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
