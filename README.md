# 🎯 Synthetic Audience Generator

Generate synthetic audience profiles with **exact demographic distribution matching**.

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key (choose one)
# For Google Gemini:
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# For Azure OpenAI:
cp .env.azure .env
# Edit .env and add your Azure OpenAI configuration
```

### 2. Run Generation

#### Parallel Processing (Recommended - Much Faster!)
```bash
# Generate with Google Gemini (default)
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/output.json

# Generate with Azure OpenAI
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/output.json --provider azure

# Generate 250 profiles with custom parallel settings
python synthetic_audience_mvp.py -i dataset/persona_input.json -o results/output.json --provider azure --batch-size 10 --max-workers 5

# Generate with specific parallel configuration
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/output.json --provider azure --parallel --batch-size 3 --max-workers 2
```

#### Sequential Processing (Slower)
```bash
# Generate profiles one at a time (slower but more stable)
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/output.json --sequential
```

### 3. Check Results
```bash
# View generated profiles
cat results/output.json
```

## 📁 Project Structure
```
📁 Synthetic-Audience/
├── 📄 synthetic_audience_mvp.py     # Main application
├── 📁 dataset/
│   ├── persona_input.json           # Full dataset (250 profiles)
│   └── small_demo_input.json        # Demo dataset (5 profiles)
├── 📁 results/                      # Generated outputs
├── 📄 requirements.txt              # Dependencies
└── 📄 .env                          # API configuration
```

## ⚙️ How It Works

1. **Input**: JSON with demographic quotas and persona templates
2. **Distribution**: Algorithm ensures exact quota compliance
3. **Generation**: AI creates behavioral content for each profile
4. **Output**: JSON with synthetic audience profiles

## 🎯 Features

- ✅ **100% Quota Compliance** - Exact demographic matching
- ✅ **High-Quality Content** - AI-generated behavioral profiles
- ✅ **Scalable** - Handles 5 to 250+ profiles
- ✅ **Production Ready** - Error handling and validation
- ✅ **Workflow Visualization** - LangGraph Mermaid diagrams

## 📊 Output Format

```json
{
  "synthetic_audience": [
    {
      "age_bucket": "GenZ",
      "gender": "Female",
      "ethnicity": "White/Caucasian",
      "about": "Personality description...",
      "goalsAndMotivations": ["Goal 1", "Goal 2", "Goal 3"],
      "frustrations": ["Frustration 1", "Frustration 2", "Frustration 3"],
      "needState": "Current motivational state",
      "occasions": "Content engagement patterns",
      "profile_id": 1
    }
  ],
  "generation_metadata": {
    "total_profiles": 5,
    "distribution_accuracy": { /* Validation results */ }
  }
}
```

## 📊 Workflow Visualization

Visualize the LangGraph workflow using Mermaid diagrams:

```bash
# View Mermaid code
python synthetic_audience_mvp.py --show-mermaid

# Save diagram as PNG
python synthetic_audience_mvp.py --save-graph workflow.png

# Display in Jupyter
python synthetic_audience_mvp.py --visualize
```

See [WORKFLOW_VISUALIZATION.md](WORKFLOW_VISUALIZATION.md) for detailed documentation.

## ⚡ Performance & Parallel Processing

### Speed Comparison
- **Sequential**: ~7 seconds per profile (one at a time)
- **Parallel**: ~2-3 seconds per profile (batch processing)
- **Speedup**: 2-3x faster with parallel processing

### Parallel Processing Configuration

#### Environment Variables
```bash
# Set in .env file or environment
PARALLEL_BATCH_SIZE=5          # Profiles per batch
MAX_WORKERS=3                  # Worker threads
CONCURRENT_REQUESTS=3          # Concurrent API calls
```

#### CLI Options
```bash
# Basic parallel processing
python synthetic_audience_mvp.py -i input.json -o output.json --parallel

# Custom batch configuration
python synthetic_audience_mvp.py -i input.json -o output.json --batch-size 10 --max-workers 5

# Sequential processing (fallback)
python synthetic_audience_mvp.py -i input.json -o output.json --sequential
```

### Performance Tips
- **Small datasets (< 10 profiles)**: Sequential might be faster due to overhead
- **Medium datasets (10-100 profiles)**: Use `--batch-size 5 --max-workers 3`
- **Large datasets (100+ profiles)**: Use `--batch-size 10 --max-workers 5`
- **API rate limits**: Reduce `CONCURRENT_REQUESTS` if you hit limits
- **Memory constraints**: Reduce `BATCH_SIZE` if you encounter memory issues

### Example Performance Test
```bash
# Run the performance comparison
python example_parallel_usage.py
```

## 🔧 API Requirements

### Google Gemini (Default)
- **Google Gemini API Key** required
- **Free Tier**: 250 requests/day
- **Recommendation**: Use `gemini-1.5-flash` for higher quotas

### Azure OpenAI (Recommended for Production)
- **Azure OpenAI Resource** required
- **Higher Rate Limits**: Typically 10,000+ requests/minute
- **Better Reliability**: Enterprise-grade SLA
- **Configuration Required**:
  ```bash
  AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
  AZURE_OPENAI_API_KEY=your_api_key
  AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
  AZURE_OPENAI_API_VERSION=2024-02-15-preview
  ```

### Provider Comparison
| Feature | Google Gemini | Azure OpenAI |
|---------|---------------|--------------|
| **Setup** | Simple API key | Resource + deployment |
| **Free Tier** | 250 requests/day | Pay-per-use |
| **Rate Limits** | Lower | Much higher |
| **Reliability** | Good | Enterprise-grade |
| **Models** | Gemini 2.5 Flash | GPT-3.5, GPT-4 |
| **Best For** | Development/Testing | Production |
