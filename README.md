# PangaeaGPT

An AI-powered platform for discovering, loading, and analysing scientific datasets from the [PANGAEA](https://www.pangaea.de/) data repository. PangaeaGPT combines a modern React frontend with a FastAPI backend that orchestrates a multi-agent LLM pipeline — including specialised oceanography, ecology, visualisation, and data-analysis agents — to answer natural-language questions about Earth system data.

<p align="center">
  <img src="img/pangaea-logo.png" width="120" alt="PangaeaGPT logo" />
</p>

---

## Features

| Capability | Description |
|---|---|
| **Semantic Dataset Search** | Natural-language queries are translated into PANGAEA API calls and ranked by an LLM search agent |
| **One-Click Loading** | Select datasets from search results and load them into a per-session workspace in seconds |
| **Multi-Agent Analysis** | A supervisor agent routes queries to specialised sub-agents: Oceanographer, Ecologist, Visualisation, DataFrame, and Writer |
| **Interactive Chat** | Real-time WebSocket chat with streaming status updates, markdown rendering, and inline plot display |
| **Dark & Light Themes** | Clean, minimalistic scientific UI with a one-click toggle — persisted to `localStorage` |
| **ERA5 / Copernicus Integration** | The Oceanographer agent can retrieve ERA5 reanalysis and Copernicus Marine data via Arraylake |
| **Sandboxed Code Execution** | All agent-generated Python code runs in an isolated REPL sandbox per session |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Frontend (React + Vite + Tailwind)                 │
│  localhost:5174                                     │
│  ┌──────────────┐  ┌────────────────────┐           │
│  │ DatasetExplorer│  │   DataAgent (WS)  │           │
│  └──────────────┘  └────────────────────┘           │
└─────────────────────────────────────────────────────┘
              │  REST / WebSocket  │
              ▼                    ▼
┌─────────────────────────────────────────────────────┐
│  Backend (FastAPI + Uvicorn)                        │
│  localhost:8000                                     │
│  ┌────────────┐  ┌──────────────────────┐           │
│  │ SessionMgr │  │ SupervisorAgent      │           │
│  └────────────┘  │  ├─ OceanographerAgt │           │
│                  │  ├─ EcologistAgent    │           │
│                  │  ├─ VisualisationAgt  │           │
│                  │  ├─ DataFrameAgent    │           │
│                  │  └─ WriterAgent       │           │
│                  └──────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- An **OpenAI API key** (set as `OPENAI_API_KEY`)

### 1. Clone & install backend

```bash
git clone https://github.com/CliDyn/pangaeaGPT.git
cd pangaeaGPT

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python setup.py             # downloads shapefiles + bathymetry data
```

### 2. Configure environment

Create a `.env` file in the project root (already in `.gitignore`):

```env
OPENAI_API_KEY=sk-...
# Optional:
LANGCHAIN_API_KEY=ls-...
ARRAYLAKE_API_KEY=...
```

### 3. Install & run frontend

```bash
cd frontend
npm install
npm run dev          # → http://localhost:5174
```

### 4. Run backend

```bash
# In a separate terminal, from the project root
source .venv/bin/activate
uvicorn api.main:app   # → http://localhost:8000
```

Open **http://localhost:5174** — the app will auto-connect to the backend.

---

## UI Overview

The interface has two main views, accessible from the left sidebar:

### Dataset Explorer
Search PANGAEA, browse result cards with metadata, preview tabular data inline, and select datasets to load into your workspace.

### Data Agent
Chat with the multi-agent pipeline. Ask questions like:
- *"What columns are in this dataset?"*
- *"Plot a map of the sampling stations"*
- *"Calculate the mean temperature by depth"*
- *"Create a species distribution model"*

The agent will plan, delegate to specialist sub-agents, execute Python code in a sandbox, and stream the results back — including inline matplotlib/cartopy plots.

### Theme Toggle
Click the ☀️ / 🌙 icon in the sidebar to switch between dark and light themes.

---

## Project Structure

```
├── api/                   # FastAPI routes and server entry point
│   ├── main.py            #   App factory, static mounts, CORS
│   └── routes/            #   REST + WebSocket endpoints
│       ├── sessions.py    #     Session lifecycle
│       ├── search.py      #     Dataset search, select, preview
│       └── agent.py       #     WebSocket agent chat
├── frontend/              # React (Vite + TypeScript + Tailwind)
│   └── src/
│       ├── App.tsx         #   Shell, sidebar, theme toggle
│       ├── index.css       #   CSS custom properties (dark/light)
│       ├── api/client.ts   #   REST client + WebSocket class
│       └── components/
│           ├── DatasetExplorer.tsx
│           ├── DataAgent.tsx
│           └── WorkspacePanel.tsx
├── src/                   # Core Python package
│   ├── agents/            #   Supervisor + specialist agents
│   ├── tools/             #   LangChain tools (REPL, ERA5, planning, …)
│   ├── search/            #   PANGAEA search + dataset download utils
│   ├── config.py          #   Env-var config loader (no hardcoded keys)
│   └── llm_factory.py     #   Model instantiation helper
├── main.py                # Session init, dataset loading, agent wiring
├── setup.py               # One-time data download script
└── requirements.txt
```

---

## Legacy Streamlit Interface

The original Streamlit frontend (`app.py`) is still available for reference:

```bash
streamlit run app.py
```

The new React + FastAPI architecture decouples frontend and backend, supports WebSocket streaming, session isolation, and theme customisation.

---

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/CliDyn/pangaeaGPT).
