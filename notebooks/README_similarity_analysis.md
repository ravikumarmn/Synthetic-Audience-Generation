# Cosine Similarity Analysis Tools

This directory contains comprehensive tools for analyzing the similarity between input personas and generated synthetic audience profiles using cosine similarity metrics.

## 📊 Similarity Scoring System

- **Score Range**: 0 to 1
- **0.0**: Identical content (perfect match)
- **1.0**: Completely different content (no similarity)

### Interpretation Guide
| Score Range | Interpretation | Meaning |
|-------------|----------------|---------|
| 0.0 - 0.2   | Very Similar   | Nearly identical content |
| 0.2 - 0.4   | Similar        | Closely related content |
| 0.4 - 0.6   | Moderately Different | Some similarities, some differences |
| 0.6 - 0.8   | Different      | Mostly different content |
| 0.8 - 1.0   | Very Different | Completely different content |

## 🛠️ Available Tools

### 1. Jupyter Notebook Analysis
**File**: `cosine_similarity_analysis.ipynb`
- Interactive analysis with detailed visualizations
- Step-by-step similarity calculations
- Pandas DataFrames for easy data exploration
- Section-wise comparison (about, goals, frustrations, etc.)

### 2. Python Script Analysis
**File**: `cosine_similarity_script.py`
- Command-line tool for batch analysis
- Automated report generation
- CSV export functionality
- Comprehensive statistics

### 3. Demo with Sample Data
**File**: `demo_similarity_with_sample_data.ipynb`
- Creates sample synthetic audience data
- Demonstrates analysis with realistic examples
- Perfect for testing and understanding the tool

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements_similarity.txt
```

### Method 1: Using Python Script
```bash
cd notebooks
python cosine_similarity_script.py
```

### Method 2: Using Jupyter Notebook
```bash
cd notebooks
jupyter notebook cosine_similarity_analysis.ipynb
```

### Method 3: Demo with Sample Data
```bash
cd notebooks
jupyter notebook demo_similarity_with_sample_data.ipynb
```

## 📁 Input/Output Files

### Input Files
- `../dataset/small_demo_input.json`: Original persona data
- `../output.json`: Generated synthetic audience profiles

### Output Files
- `../results/cosine_similarity_analysis.csv`: Detailed comparison results
- `../results/section_wise_similarity.csv`: Section-wise analysis summary

## 📊 Analysis Sections

The tool analyzes similarity across these content sections:

1. **About**: Personal background and description
2. **Goals**: Goals and motivations
3. **Frustrations**: Pain points and challenges
4. **Need State**: Current emotional/behavioral state
5. **Occasions**: Usage occasions and contexts
6. **Combined**: Overall similarity across all sections

## 🎯 Use Cases

### Quality Assessment
- Measure how well synthetic profiles match input personas
- Identify sections with high/low similarity
- Validate content generation quality

### Content Diversity Analysis
- Assess variation in generated content
- Ensure synthetic profiles aren't too similar to inputs
- Balance between accuracy and creativity

### Optimization Insights
- Identify which sections need improvement
- Guide prompt engineering for better results
- Monitor generation consistency

## 📈 Sample Output

```
🎯 SIMILARITY ANALYSIS RESULTS
==================================================
Total Comparisons: 24
Average Similarity Score: 0.3456
Score Range: 0.1234 - 0.7890

📋 Score Distribution by Interpretation:
Similar                12
Moderately Different    8
Different              4

📊 SECTION-WISE SIMILARITY ANALYSIS
==================================================
                Mean_Score  Std_Dev  Min_Score  Max_Score
about               0.234     0.123      0.100      0.456
goals               0.345     0.234      0.123      0.567
frustrations        0.456     0.345      0.234      0.678
combined            0.345     0.234      0.123      0.567
```

## 🔧 Customization

### Modify Similarity Thresholds
Edit the `get_similarity_interpretation()` function to adjust score ranges.

### Add New Sections
Extend the `extract_persona_text()` and `extract_synthetic_text()` functions to include additional content sections.

### Change Similarity Algorithm
Replace TF-IDF + Cosine Similarity with other methods like:
- Jaccard Similarity
- Semantic embeddings (BERT, etc.)
- Custom similarity metrics

## 🐛 Troubleshooting

### Empty Output File
If `output.json` is empty, the tool will:
- Create demo comparisons between input personas
- Show similarity scores of 0.0 (identical personas)
- Still generate analysis reports for testing

### Missing Dependencies
```bash
pip install pandas numpy scikit-learn jupyter matplotlib seaborn
```

### File Path Issues
Ensure you're running from the `notebooks/` directory and that input files exist in the expected locations.

## 📚 Technical Details

### Algorithm
1. **Text Preprocessing**: Lowercase, remove stop words
2. **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
3. **Similarity Calculation**: Cosine similarity between vectors
4. **Score Conversion**: Convert similarity (0-1) to dissimilarity (1-0)

### Performance
- **Processing Time**: ~1ms per comparison
- **Memory Usage**: Minimal for typical datasets
- **Scalability**: Handles hundreds of profiles efficiently

## 🤝 Contributing

To extend or improve the similarity analysis:

1. Add new similarity metrics in the `calculate_cosine_similarity()` method
2. Extend section analysis in the text extraction functions
3. Add visualization features to the notebooks
4. Implement batch processing for large datasets

## 📄 License

This tool is part of the Synthetic Audience Generator project and follows the same licensing terms.
