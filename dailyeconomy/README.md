# Daily Economy Report Generator

This system processes daily FRED (Federal Reserve Economic Data) measurements to generate a comprehensive economy report with visualizations, trend analyses, and AI-generated insights using Ollama with the deepseek-r1:70b model.

## Features

- Loads and processes daily FRED economic data measurements
- Generates visualizations for each economic indicator
- Analyzes recent trends and compares with historical economic downturns
- Calculates recession risk indicators based on key economic metrics
- Generates AI-powered insights using Ollama with deepseek-r1:70b model
- Creates a comprehensive HTML report with all analyses

## Requirements

- Python 3.8 or higher
- Ollama running locally with deepseek-r1:70b model
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository or download the source code
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running with the deepseek-r1:70b model:

```bash
# If you need to install and run Ollama, follow instructions at https://ollama.ai/
ollama pull deepseek-r1:70b
```

## Usage

Run the main script with the following command:

```bash
python daily_economy_report.py --date YYYYMMDD --data-dir /path/to/data --context-file /path/to/context.txt --output-dir /path/to/output --verbose
```

Parameters:
- `--date` - Date in YYYYMMDD format for the report (default: today)
- `--data-dir` - Directory containing FRED data (default: from config.py)
- `--context-file` - Path to the context file (default: from config.py)
- `--output-dir` - Directory to save the report (default: from config.py)
- `--verbose` - Enable verbose logging

### Example usage:

```bash
# Generate report using sample data with verbose output
python daily_economy_report.py --date 20240501 --verbose

# Generate report using custom data locations 
python daily_economy_report.py --date 20240501 --data-dir /path/to/data --output-dir /path/to/output
```

### Testing the LLM Integration

If you encounter issues with the LLM integration, ensure:

1. Ollama is running on your system (`ollama serve` command if not running as a service)
2. The deepseek-r1:70b model is available (install with `ollama pull deepseek-r1:70b`)
3. The API endpoint in config.py is set correctly (default: http://localhost:11434/api/generate)

You can test with a small data sample to verify LLM insights are generating correctly:

```bash
python daily_economy_report.py --date 20240501 --verbose
```

## Data Format

### FRED Data Files

The system expects FRED data files in the following format:

```
#MEASUREMENT:GDP
#HEADERS:date,value
2023-01-01,26208.0
2023-02-01,26429.6
...

#MEASUREMENT:UNRATE
#HEADERS:date,value
2023-01-01,3.4
2023-02-01,3.6
...
```

### Context File (fred_context.txt)

The context file should provide additional information about each measurement:

```
MEASUREMENT:GDP
title: Gross Domestic Product
description: Total value of goods and services produced
frequency: quarterly
units: Billions of dollars

MEASUREMENT:UNRATE
title: Unemployment Rate
description: Percentage of labor force that is unemployed
frequency: monthly
units: %
```

## Output

The system generates:

1. An HTML report with visualizations and analyses
2. Individual visualization PNG files for each measurement
3. A log file with processing information
4. A metadata file with summary information

## Project Structure

```
/dailyeconomy/
├── config.py                # Configuration settings
├── daily_economy_report.py  # Main script
├── modules/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading functions
│   ├── visualizer.py        # Visualization functions
│   ├── analyzer.py          # Analysis functions
│   ├── llm_insights.py      # LLM integration
│   └── report_generator.py  # Report generation
├── templates/              # HTML/Report templates
│   └── report_template.html
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Helper utilities
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 