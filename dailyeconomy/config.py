# Paths
DATA_DIR = "/Users/zeno.su/Desktop/dailyeconomy"  # Original data directory
CONTEXT_FILE = f"{DATA_DIR}/fred_context.txt"     # Context file
OUTPUT_DIR = "/Users/zeno.su/Desktop/dailyeconomy/reports"  # Reports output

# LLM Configuration
OLLAMA_MODEL = "deepseek-r1:70b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API

# Analysis Parameters
RECESSION_RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9
}

# Historical Events
HISTORICAL_EVENTS = {
    "financial_crisis": "2008-09-01",
    "covid_drop": "2020-03-01"
}

# Visualization Settings
VIZ_WIDTH = 10
VIZ_HEIGHT = 6
VIZ_DPI = 100 