# RAG-Based Duplicate Detection System

## Overview

The RAG (Retrieval-Augmented Generation) duplicate detection system prevents the generation of similar behavioral content across synthetic audience profiles. It uses section-wise similarity detection with LangChain's InMemoryVectorStore and sentence transformers for embeddings.

## Key Features

### 🎯 Section-Wise Detection
- **Granular Analysis**: Checks similarity at the section level (about, goals, frustrations, needState, occasions)
- **Intelligent Reuse**: Reuses similar existing content instead of generating duplicates
- **Selective Generation**: Only generates new content for sections that aren't similar to existing ones

### 🔧 Configurable Thresholds
- **Global Threshold**: Set overall similarity threshold (default: 0.85)
- **Section-Specific**: Configure different thresholds per section type
- **Dynamic Adjustment**: Change thresholds at runtime

### 🚀 Performance Optimized
- **In-Memory Storage**: Fast similarity searches using InMemoryVectorStore
- **Efficient Embeddings**: Uses lightweight sentence transformers (all-MiniLM-L6-v2)
- **Batch Processing**: Supports parallel processing workflows

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLM Content   │    │  RAG Duplicate   │    │  Vector Stores  │
│   Generator     │───▶│   Detector       │───▶│  (per section)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Similarity Check │
                       │ & Content Reuse  │
                       └──────────────────┘
```

## Components

### 1. RAGDuplicateDetector
Main class for duplicate detection and content management.

```python
from rag_duplicate_detector import RAGDuplicateDetector

detector = RAGDuplicateDetector(
    similarity_threshold=0.85,
    embedding_model="all-MiniLM-L6-v2"
)
```

### 2. RAGEnhancedLLMGenerator
Enhanced LLM generator with integrated RAG capabilities.

```python
from rag_enhanced_generator import RAGEnhancedLLMGenerator

generator = RAGEnhancedLLMGenerator(
    provider="google",  # or "azure"
    enable_rag=True,
    similarity_threshold=0.85
)
```

## Usage Examples

### Basic RAG Detection

```python
# Initialize detector
detector = RAGDuplicateDetector(similarity_threshold=0.8)

# Sample behavioral content
content = {
    "about": "Passionate about digital marketing and social media strategy",
    "goalsAndMotivations": [
        "Build a strong personal brand online",
        "Learn advanced analytics tools"
    ],
    "frustrations": [
        "Difficulty measuring ROI on social campaigns",
        "Keeping up with platform algorithm changes"
    ],
    "needState": "Seeking tools to optimize social media performance",
    "occasions": "Active during evening hours for content planning"
}

# Check and upsert content
final_content, similarity_results = detector.upsert_behavioral_content(
    content, 
    profile_id=1
)

# Check results
for section, result in similarity_results.items():
    if result.is_similar:
        print(f"{section}: REUSED (similarity: {result.similarity_score:.3f})")
    else:
        print(f"{section}: NEW")
```

### Enhanced LLM Generation

```python
# Initialize enhanced generator
generator = RAGEnhancedLLMGenerator(enable_rag=True)

# Generate content with RAG
result = generator.generate_content_with_rag(
    templates=processing_templates,
    profile_id=1
)

print(f"Content generated: {result.was_generated}")
print(f"Generation attempts: {result.generation_attempts}")

# Access the behavioral content
behavioral_content = result.content
print(f"About: {behavioral_content.about}")
```

### Backward Compatibility

```python
# Drop-in replacement for existing LLMContentGenerator
from rag_enhanced_generator import LLMContentGenerator

# Works exactly like the original, but with RAG enabled
generator = LLMContentGenerator(provider="google", enable_rag=True)
content = generator.generate_content(templates)
```

## Configuration

### Environment Variables

```bash
# Enable/disable RAG
RAG_ENABLED=true

# Global similarity threshold
RAG_SIMILARITY_THRESHOLD=0.85

# Embedding model
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Section-specific thresholds (optional)
RAG_ABOUT_THRESHOLD=0.85
RAG_GOALS_THRESHOLD=0.80
RAG_FRUSTRATIONS_THRESHOLD=0.80
RAG_NEED_STATE_THRESHOLD=0.85
RAG_OCCASIONS_THRESHOLD=0.75

# Performance settings
RAG_MAX_GENERATION_ATTEMPTS=3
RAG_ENABLE_STATS_LOGGING=true
```

### Similarity Thresholds Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.95+ | Very strict - only nearly identical content | High uniqueness requirements |
| 0.85-0.94 | Strict - similar content detected | Balanced duplicate prevention |
| 0.70-0.84 | Moderate - broader similarity detection | Aggressive deduplication |
| 0.50-0.69 | Lenient - loose similarity matching | Maximum content reuse |

## Section Types

The system handles five behavioral content sections:

1. **About** (`about`): String content describing interests and behaviors
2. **Goals** (`goalsAndMotivations`): List of achievement-oriented goals
3. **Frustrations** (`frustrations`): List of challenges and pain points
4. **Need State** (`needState`): String describing current psychological state
5. **Occasions** (`occasions`): String describing engagement patterns

## Performance Characteristics

### Benchmarks
- **Processing Time**: ~50-100ms per profile (with existing content)
- **Memory Usage**: ~10-20MB for 1000 profiles
- **Similarity Search**: Sub-millisecond for typical datasets

### Optimization Tips
1. **Adjust Thresholds**: Lower thresholds increase reuse but may reduce uniqueness
2. **Monitor Statistics**: Use `get_rag_stats()` to track performance
3. **Clear Memory**: Periodically clear vector stores for long-running processes
4. **Batch Processing**: Process multiple profiles in batches for better performance

## Integration with Existing Workflow

### LangGraph Integration

```python
# In your LangGraph workflow nodes
def rag_enhanced_llm_node(state: SyntheticAudienceState):
    """Enhanced LLM node with RAG duplicate detection."""
    generator = RAGEnhancedLLMGenerator(
        provider=state.get("provider", "google"),
        enable_rag=True
    )
    
    templates = state["templates"]
    profile_id = state.get("current_profile_id")
    
    # Generate with RAG
    result = generator.generate_content_with_rag(
        templates, 
        profile_id=profile_id
    )
    
    # Update state
    state["behavioral_content"] = result.content
    state["rag_similarity_results"] = result.similarity_results
    state["content_was_generated"] = result.was_generated
    
    return state
```

### Parallel Processing Support

```python
# RAG works with existing parallel processing
class RAGParallelBatchProcessor:
    def __init__(self):
        # Shared RAG detector across workers
        self.rag_detector = RAGDuplicateDetector()
    
    async def process_batch(self, profiles):
        # Process profiles with shared RAG memory
        results = []
        for profile in profiles:
            result = await self.process_single_profile(profile)
            results.append(result)
        return results
```

## Monitoring and Statistics

### RAG Statistics

```python
# Get comprehensive statistics
stats = detector.get_stats()

print(f"Similarity threshold: {stats['similarity_threshold']}")
print(f"Embedding model: {stats['embedding_model']}")

for section, section_stats in stats['section_stats'].items():
    print(f"{section}: {section_stats['document_count']} unique items")
```

### Similarity Results Analysis

```python
# Analyze similarity patterns
similarity_results = result.similarity_results

reused_sections = [
    section for section, result in similarity_results.items()
    if result.is_similar
]

print(f"Reused sections: {reused_sections}")
print(f"Uniqueness ratio: {len(reused_sections)/len(similarity_results):.2%}")
```

## Testing

### Run Test Suite

```bash
# Run comprehensive tests
python test_rag_system.py

# Run integration examples
python example_rag_integration.py
```

### Test Categories
1. **Basic Functionality**: Core RAG operations
2. **Similarity Thresholds**: Different threshold behaviors
3. **Section Isolation**: Independent section processing
4. **List Content Handling**: Goals and frustrations processing
5. **Memory Management**: Statistics and clearing
6. **Edge Cases**: Error handling and malformed data
7. **Performance**: Processing speed and memory usage

## Troubleshooting

### Common Issues

**1. High Memory Usage**
```python
# Clear RAG memory periodically
detector.clear_all()

# Or clear specific sections
detector.clear_section(SectionType.ABOUT)
```

**2. Too Many Duplicates Detected**
```python
# Lower similarity threshold
detector.similarity_threshold = 0.7

# Or disable RAG temporarily
generator = RAGEnhancedLLMGenerator(enable_rag=False)
```

**3. Slow Performance**
```python
# Use faster embedding model
detector = RAGDuplicateDetector(
    embedding_model="all-MiniLM-L6-v2"  # Faster than all-mpnet-base-v2
)
```

**4. Import Errors**
```bash
# Install required dependencies
pip install langchain-community sentence-transformers
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed RAG logging
detector = RAGDuplicateDetector(similarity_threshold=0.85)
```

## Best Practices

### 1. Threshold Selection
- Start with 0.85 for balanced behavior
- Adjust based on uniqueness requirements
- Monitor reuse rates and adjust accordingly

### 2. Memory Management
- Clear memory after processing large batches
- Monitor memory usage in long-running processes
- Use section-specific clearing for targeted cleanup

### 3. Performance Optimization
- Use appropriate embedding models for your use case
- Batch process multiple profiles when possible
- Monitor processing times and adjust thresholds

### 4. Integration Strategy
- Enable RAG gradually in existing workflows
- Test thoroughly with your specific data patterns
- Monitor similarity patterns and adjust configuration

## Future Enhancements

### Planned Features
1. **Persistent Storage**: Save RAG memory to disk
2. **Advanced Similarity Metrics**: Custom similarity functions
3. **Hierarchical Clustering**: Group similar profiles
4. **Real-time Analytics**: Live similarity monitoring
5. **API Integration**: REST API for RAG operations

### Contributing
- Report issues and feature requests
- Submit performance optimizations
- Add new similarity metrics
- Improve documentation and examples

---

For more examples and advanced usage, see:
- `example_rag_integration.py` - Comprehensive integration examples
- `test_rag_system.py` - Test suite and validation
- `.env.rag` - Configuration template
