# Custom UI Styles
CUSTOM_UI = """
<style>
    :root {
        /* Main colors (lighter versions) */
        --primary-teal: rgb(67, 163, 151);          
        --primary-teal-light: rgba(67, 163, 151, 0.1);
        --neutral-gray: rgb(220, 220, 220);         
        --primary-blue: rgb(82, 142, 198);          
        --primary-blue-light: rgba(82, 142, 198, 0.1);
        --dark-blue: rgb(65, 105, 145);             
        --cream-bg: rgba(218,6,18, 0.05);             
        --accent-red: rgb(235, 108, 108);           
        --accent-red-light: rgba(235, 108, 108, 0.1);
        --white: #ffffff;
        --text-dark: rgb(60, 60, 60);
    }

    /* Thinking Log Styles */
    .thinking-log {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 10px;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        border-left: 3px solid var(--primary-blue);
    }

    .thinking-log .entry {
        margin-bottom: 5px;
        padding: 5px;
        border-bottom: 1px solid #eee;
    }

    .thinking-log .agent-name {
        color: var(--primary-blue);
        font-weight: bold;
    }

    .thinking-log .tool-name {
        color: var(--primary-teal);
        font-style: italic;
    }

    .thinking-log .timestamp {
        color: #888;
        font-size: 0.8em;
    }
    /* Global Styles */
    .stApp {
        background-color: var(--white);
        color: var(--text-dark);
        font-family: "Roboto", sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--dark-blue);
        border-bottom: 2px solid var(--primary-teal-light);
        padding-bottom: 0.2em;
        background: linear-gradient(to right, var(--white) 0%, var(--primary-teal-light) 100%);
        background-clip: text;
        -webkit-background-clip: text;
        color: var(--dark-blue); /* re-apply color since gradient would show through transparent text */
        text-shadow: 0 1px 1px rgba(0,0,0,0.1);
        font-weight: 600;
    }

    /* Chat Messages */
    .stChatMessage {
        background: var(--primary-blue-light);
        border-left: 3px solid var(--primary-blue);
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 0.5em;
        transition: box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Buttons */
    .stButton > button {
        background: var(--white);
        color: var(--primary-teal);
        border: 1px solid var(--primary-teal);
        border-radius: 4px;
        transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: var(--primary-teal);
        color: var(--white);
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }

    /* Fixed Button (if any) */
    .fixed-button button {
        background: var(--primary-blue);
        color: var(--white);
        border-radius: 4px;
    }

    /* Sidebar */
    [data-testid=stSidebar] {
        background-color: var(--cream-bg);
        color: var(--text-dark);
        border-right: 1px solid var(--neutral-gray);
    }
    [data-testid=stSidebar] .stSelectbox label {
        color: var(--text-dark);
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background: var(--white);
        border: 1px solid var(--neutral-gray);
        border-radius: 4px;
        padding: 0.4em;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 1px var(--primary-blue-light);
    }

    /* Alerts and Messages */
    .stAlert {
        background: var(--primary-teal-light);
        border-left: 3px solid var(--primary-teal);
        border-radius: 4px;
        padding: 0.5em;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stAlert.error {
        background: var(--accent-red-light);
        border-left: 3px solid var(--accent-red);
    }

    /* DataFrame */
    .stDataFrame {
        border: 1px solid var(--neutral-gray);
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--white);
        border: 1px solid var(--neutral-gray);
        border-radius: 4px;
        font-weight: 500;
    }
    .streamlit-expanderHeader:hover {
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Checkbox */
    .stCheckbox > label > div[role="checkbox"] {
        border-color: var(--primary-teal);
        border-radius: 3px;
        transition: background 0.2s ease;
    }
    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background-color: var(--primary-teal);
    }

    /* Search bar */
    .stSearchInput > div > div > input {
        border: 1px solid var(--primary-blue);
        border-radius: 4px;
    }
    .stSearchInput > div > div > input:focus {
        border-color: var(--primary-teal);
        box-shadow: 0 0 0 1px var(--primary-teal-light);
    }

    /* Secondary buttons */
    [data-testid="stButton"] > button[kind="secondary"] {
        background: var(--white);
        color: var(--primary-blue);
        border: 1px solid var(--primary-blue);
        border-radius: 4px;
        transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stButton"] > button[kind="secondary"]:hover {
        background: var(--primary-blue);
        color: var(--white);
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }

    /* Delete or cancel actions */
    .delete-button > button {
        color: var(--accent-red);
        border-color: var(--accent-red);
        border-radius: 4px;
        transition: background 0.2s ease, color 0.2s ease;
    }
    .delete-button > button:hover {
        background: var(--accent-red);
        color: var(--white);
    }
    
    /* Search Mode Toggle */
    [data-testid="stRadio"] > div[role="radiogroup"] {
        background-color: var(--primary-blue-light);
        border-radius: 8px;
        padding: 8px;
        gap: 10px;
    }
    
    [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: var(--white);
        border: 1px solid var(--primary-blue);
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: var(--primary-blue);
        color: var(--white);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
"""


