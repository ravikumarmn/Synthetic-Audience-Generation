# 🚀 Quick Start Guide

## 📋 What You Need

1. **Python 3.8+** installed
2. **Google Gemini API Key** ([Get one here](https://ai.google.dev/))
3. **5 minutes** to set up

---

## ⚡ 3-Step Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Key
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key:
# GOOGLE_API_KEY=your_api_key_here
```

### Step 3: Run Generation
```bash
# Option A: Use the example script (recommended for beginners)
python example_usage.py

# Option B: Run directly (for advanced users)
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/output.json
```

---

## 📁 What's in the Project

```
📁 Synthetic-Audience/
├── 📄 synthetic_audience_mvp.py     # ⭐ Main application (the core!)
├── 📄 example_usage.py              # 🎯 Example script to get started
├── 📁 dataset/
│   ├── persona_input.json           # 📊 Full dataset (250 profiles)
│   └── small_demo_input.json        # 🧪 Demo dataset (5 profiles)
├── 📁 results/
│   └── audience_5_profiles.json     # ✅ Example output
├── 📄 requirements.txt              # 📦 Dependencies
├── 📄 .env                          # 🔑 Your API key (create this)
└── 📄 README.md                     # 📖 Full documentation
```

---

## 🎯 Example Commands

### Generate 5 Profiles (Quick Test)
```bash
python synthetic_audience_mvp.py -i dataset/small_demo_input.json -o results/test_output.json
```

### Generate 250 Profiles (Full Dataset)
```bash
python synthetic_audience_mvp.py -i dataset/persona_input.json -o results/full_output.json
```

### Check Your Results
```bash
# View the generated JSON
cat results/test_output.json

# Or use a JSON viewer
python -m json.tool results/test_output.json
```

---

## 🔧 Troubleshooting

### "Quota Exceeded" Error
- **Wait 24 hours** for free tier reset, OR
- **Switch to gemini-1.5-flash** in your .env file, OR
- **Upgrade to paid tier** for unlimited usage

### "API Key Not Found" Error
- Make sure you created the `.env` file
- Check that your API key is correct
- Ensure no extra spaces in the .env file

### "Input File Not Found" Error
- Check that the dataset folder exists
- Make sure you're in the right directory
- Use the full path if needed

---

## 🎉 Success!

If everything works, you'll see:
1. ✅ Generation progress (1/5, 2/5, etc.)
2. ✅ "Generation completed successfully!"
3. ✅ JSON file created in results/ folder

**Your synthetic audience is ready to use!** 🚀
