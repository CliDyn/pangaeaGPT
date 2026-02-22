import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';
const WS_BASE_URL = 'ws://localhost:8000/api';

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
    }
};

// --- WebSocket Service ---
type WSMessageCallback = (msg: unknown) => void;

export class AgentWebSocket {
    private ws: WebSocket | null = null;
    private onMessage: WSMessageCallback;
    private onError: WSMessageCallback;
    private onClose: () => void;
    private url: string;

    constructor(
        sessionId: string,
        onMessage: WSMessageCallback,
        onError: WSMessageCallback,
        onClose: () => void
    ) {
        this.url = `${WS_BASE_URL}/sessions/${sessionId}/agent/ws`;
        this.onMessage = onMessage;
        this.onError = onError;
        this.onClose = onClose;
    }

    connect() {
        this.ws = new WebSocket(this.url);

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
            this.onClose();
        };
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
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
