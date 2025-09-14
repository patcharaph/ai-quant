# 📋 Log Files Locations

## 🔍 Where to Find Logs

### 1. **LLM Logs** (AI Advisory)
- **Location**: `llm_logs/` directory
- **Format**: JSONL files (one per day)
- **Naming**: `llm_interactions_YYYYMMDD.jsonl`
- **Content**: AI advisory interactions, API calls, responses

### 2. **Data Cache** (Stock Data)
- **Location**: `data_cache/` directory  
- **Format**: Parquet files
- **Naming**: `{SYMBOL}_{START_DATE}_{END_DATE}_{INTERVAL}.parquet`
- **Content**: Cached stock price data

### 3. **Application Logs** (System)
- **Location**: Console output (when running)
- **Format**: Text logs with timestamps
- **Content**: System status, errors, warnings

### 4. **Streamlit Logs** (Web Interface)
- **Location**: Streamlit's internal logging
- **Format**: Streamlit log format
- **Content**: Web app errors, warnings

## 🛠️ How to Access Logs

### View LLM Logs
```bash
# List LLM log files
ls llm_logs/

# View latest LLM log
cat llm_logs/llm_interactions_$(date +%Y%m%d).jsonl

# View all LLM logs
find llm_logs/ -name "*.jsonl" -exec cat {} \;
```

### View Data Cache
```bash
# List cached data files
ls data_cache/

# View specific cached data
python -c "import pandas as pd; print(pd.read_parquet('data_cache/PTT.BK_2024-01-01_2024-12-31_1d.parquet').head())"
```

### View Application Logs
```bash
# Run app and capture logs
python app.py 2>&1 | tee app_logs.txt

# Or run with Streamlit
streamlit run app.py 2>&1 | tee streamlit_logs.txt
```

## 📊 Log Analysis

### LLM Log Analysis
```python
import json
import pandas as pd

# Load LLM logs
with open('llm_logs/llm_interactions_20240914.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# Convert to DataFrame for analysis
df = pd.DataFrame(logs)
print(df.head())
print(f"Total interactions: {len(df)}")
```

### Data Cache Analysis
```python
import pandas as pd
import os

# List all cached files
cache_files = [f for f in os.listdir('data_cache/') if f.endswith('.parquet')]
print(f"Cached files: {cache_files}")

# Analyze cache usage
for file in cache_files:
    data = pd.read_parquet(f'data_cache/{file}')
    print(f"{file}: {len(data)} rows, {data.index[0]} to {data.index[-1]}")
```

## 🔧 Troubleshooting

### If No Logs Found
1. **LLM Logs**: Run the app and use LLM features to generate logs
2. **Data Cache**: Fetch stock data to create cache files
3. **Application Logs**: Check console output when running the app

### Log File Permissions
- Ensure write permissions for log directories
- Check disk space for log file storage

### Log Rotation
- LLM logs: One file per day (automatic)
- Data cache: Manual cleanup recommended
- Application logs: Based on console output

## 📝 Log Levels

- **INFO**: General information
- **WARNING**: Potential issues
- **ERROR**: Errors that don't stop execution
- **CRITICAL**: Fatal errors

## 🎯 Quick Commands

```bash
# Find all log files
find . -name "*.log" -o -name "*.jsonl" -o -name "*.parquet"

# View recent LLM activity
tail -f llm_logs/llm_interactions_$(date +%Y%m%d).jsonl

# Check data cache size
du -sh data_cache/

# Monitor app logs in real-time
streamlit run app.py --logger.level debug
```
