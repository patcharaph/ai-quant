# Environment Setup Guide

This guide explains how to set up environment variables for the AI Quant Stock Prediction System across different deployment environments.

## Environment Strategy

The system uses a three-tier environment variable management approach:

### 🏠 Local Development
- **File**: `.env` (ignored by `.gitignore`)
- **Purpose**: Personal development environment
- **Setup**: Copy from `.env.example` and customize

### 📦 Repository
- **File**: `.env.example` (committed to version control)
- **Purpose**: Template for other developers
- **Setup**: Contains all required variables with placeholder values

### 🚀 Production
- **Source**: Platform environment variables
- **Purpose**: Secure production deployment
- **Setup**: Set directly in deployment platform (Heroku, AWS, etc.)

## Quick Start

### 1. Create Local Environment File

```bash
# Option 1: Use the environment manager
python env_manager.py create-env

# Option 2: Manual copy
cp .env.example .env
```

### 2. Configure Your API Keys

Edit the `.env` file with your actual API keys:

```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_actual_api_key_here
OPENROUTER_MODEL=openrouter/auto
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional: Custom model settings
MAX_TOKENS=500
TEMPERATURE=0.7
```

### 3. Verify Configuration

```bash
# Check environment status
python env_manager.py info

# Test configuration loading
python env_manager.py
```

## Environment Variables Reference

### Required Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | *Required* | `sk-or-v1-...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENROUTER_MODEL` | AI model to use | `openrouter/auto` | `openai/gpt-4o-mini` |
| `OPENROUTER_BASE_URL` | API base URL | `https://openrouter.ai/api/v1` | `https://openrouter.ai/api/v1` |
| `MAX_TOKENS` | Maximum tokens per response | `500` | `1000` |
| `TEMPERATURE` | Model temperature (0.0-1.0) | `0.7` | `0.5` |

### Available Models

The system supports various AI models through OpenRouter:

- `openrouter/auto` - Auto-select best model (recommended)
- `openai/gpt-4o-mini` - GPT-4o Mini (cost-effective)
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo (fast)
- `anthropic/claude-3-haiku` - Claude 3 Haiku (efficient)
- `anthropic/claude-3-sonnet` - Claude 3 Sonnet (balanced)
- `google/gemini-pro` - Gemini Pro (Google's model)
- `meta-llama/llama-3.1-8b-instruct` - Llama 3.1 8B (open source)
- `mistralai/mixtral-8x7b-instruct` - Mixtral 8x7B (multilingual)
- `qwen/qwen-2.5-7b-instruct` - Qwen 2.5 7B (efficient)

## Deployment Environments

### Local Development

1. **Create `.env` file**:
   ```bash
   python env_manager.py create-env
   ```

2. **Edit with your keys**:
   ```bash
   # Edit .env file
   OPENROUTER_API_KEY=your_actual_key_here
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Production Deployment

#### Heroku

Set environment variables in Heroku dashboard or CLI:

```bash
heroku config:set OPENROUTER_API_KEY=your_actual_key_here
heroku config:set OPENROUTER_MODEL=openrouter/auto
heroku config:set MAX_TOKENS=500
heroku config:set TEMPERATURE=0.7
```

#### AWS (Elastic Beanstalk)

1. Go to AWS Elastic Beanstalk console
2. Select your environment
3. Go to Configuration → Software
4. Add environment variables in the Environment Properties section

#### Docker

Create a `.env` file or use environment variables:

```dockerfile
# Dockerfile
ENV OPENROUTER_API_KEY=your_actual_key_here
ENV OPENROUTER_MODEL=openrouter/auto
```

Or use docker-compose:

```yaml
# docker-compose.yml
services:
  ai-quant:
    environment:
      - OPENROUTER_API_KEY=your_actual_key_here
      - OPENROUTER_MODEL=openrouter/auto
```

## Security Best Practices

### 🔒 API Key Security

1. **Never commit `.env` files** to version control
2. **Use strong, unique API keys**
3. **Rotate keys regularly**
4. **Use different keys for different environments**

### 🛡️ Environment Isolation

1. **Separate keys** for development, staging, and production
2. **Monitor API usage** to detect unauthorized access
3. **Set up alerts** for unusual activity
4. **Use environment-specific configurations**

### 📋 Checklist

- [ ] `.env` file is in `.gitignore`
- [ ] `.env.example` contains all required variables
- [ ] API keys are not hardcoded in source code
- [ ] Production uses platform environment variables
- [ ] Different keys for different environments
- [ ] Regular key rotation schedule

## Troubleshooting

### Common Issues

#### 1. "OPENROUTER_API_KEY is required" Error

**Problem**: API key not found or invalid.

**Solution**:
```bash
# Check if .env file exists
python env_manager.py info

# Create .env file if missing
python env_manager.py create-env

# Verify API key is set
python env_manager.py
```

#### 2. "No .env or .env.example file found" Warning

**Problem**: Missing environment files.

**Solution**:
```bash
# Ensure .env.example exists (should be in repo)
ls -la .env.example

# Create .env from example
python env_manager.py create-env
```

#### 3. API Connection Errors

**Problem**: Invalid API key or network issues.

**Solution**:
1. Verify API key is correct
2. Check OpenRouter account status
3. Verify network connectivity
4. Check API rate limits

### Debug Commands

```bash
# Check environment status
python env_manager.py info

# Test configuration loading
python env_manager.py

# Create local environment
python env_manager.py create-env
```

## Environment Manager API

The `env_manager.py` module provides a programmatic interface:

```python
from env_manager import get_env_config, create_local_env, get_environment_info

# Get configuration
config = get_env_config()
api_key = config.OPENROUTER_API_KEY

# Create local environment
success = create_local_env()

# Get environment info
info = get_environment_info()
```

## Migration from Old System

If you're upgrading from the old environment system:

1. **Backup your current `.env` file** (if you have one)
2. **Run the migration**:
   ```bash
   python env_manager.py create-env
   ```
3. **Update your API keys** in the new `.env` file
4. **Test the application**:
   ```bash
   streamlit run app.py
   ```

## Support

If you encounter issues with environment setup:

1. Check this documentation
2. Run `python env_manager.py info` for diagnostics
3. Verify your API keys are correct
4. Check the application logs for specific error messages

---

**Note**: This environment management system is designed to be secure, flexible, and easy to use across different deployment scenarios. Always follow security best practices when handling API keys and sensitive configuration data.
