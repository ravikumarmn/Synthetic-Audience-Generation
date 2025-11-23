# 🎯 Synthetic Audience Generator

Generate synthetic audience profiles with **exact demographic distribution matching**.

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Run Generation
```bash
# Generate 5 profiles (small demo)
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/output.json

# Generate 250 profiles (full dataset)
python synthetic_audience_mvp.py -i dataset/persona_input.json -o results/output.json
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

## 🔧 API Requirements

- **Google Gemini API Key** required
- **Free Tier**: 250 requests/day
- **Recommendation**: Use `gemini-1.5-flash` for higher quotas
