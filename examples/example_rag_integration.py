#!/usr/bin/env python3
"""
Example RAG Integration for Synthetic Audience Generator

This script demonstrates how to integrate the RAG-based duplicate detection
system into the existing synthetic audience generation workflow.
"""

import json
import os
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

# Import the RAG-enhanced components
from simple_rag_detector import SimpleRAGDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def create_sample_templates() -> Dict[str, List[str]]:
    """Create sample processing templates for testing."""
    return {
        "about_templates": [
            "Passionate about digital marketing and social media strategy",
            "Enthusiastic about creative content and brand storytelling",
            "Focused on data-driven marketing approaches",
            "Dedicated to building authentic online communities",
            "Committed to sustainable and ethical business practices",
        ],
        "goals_templates": [
            "Build a strong personal brand online",
            "Learn advanced analytics tools",
            "Grow professional network",
            "Master video content creation",
            "Develop expertise in SEO",
            "Launch a successful podcast",
            "Create viral marketing campaigns",
            "Build an engaged email list",
        ],
        "frustrations_templates": [
            "Difficulty measuring ROI on social campaigns",
            "Keeping up with platform algorithm changes",
            "Limited budget for paid advertising",
            "Time constraints for content creation",
            "Lack of quality analytics tools",
            "Inconsistent brand messaging across platforms",
            "Difficulty finding authentic influencers",
            "Managing multiple social media accounts",
        ],
        "need_state_templates": [
            "Seeking tools and strategies to optimize social media performance",
            "Looking for cost-effective marketing automation solutions",
            "Needing better analytics and reporting capabilities",
            "Searching for authentic ways to connect with target audience",
            "Requiring streamlined content creation workflows",
        ],
        "occasions_templates": [
            "Active during evening hours and weekends for content planning",
            "Most active during morning hours and lunch breaks",
            "Engages primarily on weekdays during business hours",
            "Peak activity during commute times and late evenings",
            "Concentrated usage during weekend planning sessions",
        ],
    }


def demonstrate_basic_rag_usage():
    """Demonstrate basic RAG duplicate detection usage."""
    print("=== Basic RAG Usage Demo ===")

    # Initialize RAG detector
    detector = SimpleRAGDetector(similarity_threshold=0.8)

    # Sample behavioral content
    content1 = {
        "about": "Passionate about digital marketing and social media strategy",
        "goalsAndMotivations": [
            "Build a strong personal brand online",
            "Learn advanced analytics tools",
        ],
        "frustrations": [
            "Difficulty measuring ROI on social campaigns",
            "Keeping up with platform algorithm changes",
        ],
        "needState": "Seeking tools to optimize social media performance",
        "occasions": "Active during evening hours for content planning",
    }

    content2 = {
        "about": "Enthusiastic about digital marketing and social media tactics",  # Similar
        "goalsAndMotivations": [
            "Develop expertise in data analysis",  # Different
            "Create engaging video content",
        ],
        "frustrations": [
            "Limited budget for paid advertising",  # Different
            "Time constraints for content creation",
        ],
        "needState": "Looking for cost-effective automation solutions",  # Different
        "occasions": "Most active during morning hours and lunch breaks",  # Different
    }

    # Test first content (should be added)
    print("\n1. Adding first behavioral content...")
    final1, results1 = detector.upsert_behavioral_content(content1, profile_id=1)

    for section, result in results1.items():
        print(
            f"   {section}: Similar={result.is_similar}, Score={result.similarity_score:.3f}"
        )

    # Test second content (should detect similarity in 'about' section)
    print("\n2. Adding second behavioral content...")
    final2, results2 = detector.upsert_behavioral_content(content2, profile_id=2)

    for section, result in results2.items():
        print(
            f"   {section}: Similar={result.is_similar}, Score={result.similarity_score:.3f}"
        )
        if result.is_similar:
            print(f"      -> Reused: {result.similar_content[:50]}...")

    # Show statistics
    print("\n3. RAG Statistics:")
    stats = detector.get_stats()
    print(json.dumps(stats, indent=2))


def demonstrate_enhanced_generator():
    """Demonstrate the RAG-enhanced LLM generator."""
    print("\n\n=== RAG-Enhanced Generator Demo ===")
    print("⚠️  LLM generation demo requires API keys.")
    print("   Set GOOGLE_API_KEY or Azure OpenAI credentials to test LLM integration.")
    print("   For now, showing basic RAG functionality only.")


def demonstrate_similarity_thresholds():
    """Demonstrate different similarity thresholds."""
    print("\n\n=== Similarity Threshold Demo ===")

    # Test content with varying similarity levels
    base_content = {
        "about": "Passionate about digital marketing and social media strategy",
        "goalsAndMotivations": ["Build a strong personal brand online"],
        "frustrations": ["Difficulty measuring ROI on social campaigns"],
        "needState": "Seeking tools to optimize social media performance",
        "occasions": "Active during evening hours for content planning",
    }

    similar_content = {
        "about": "Enthusiastic about digital marketing and social media tactics",  # Very similar
        "goalsAndMotivations": ["Build a strong personal brand online"],  # Identical
        "frustrations": ["Difficulty measuring ROI on social campaigns"],  # Identical
        "needState": "Looking for tools to optimize social media performance",  # Very similar
        "occasions": "Active during evening hours for content planning",  # Identical
    }

    # Test different thresholds
    thresholds = [0.6, 0.8, 0.95]

    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")

        detector = SimpleRAGDetector(similarity_threshold=threshold)

        # Add base content
        detector.upsert_behavioral_content(base_content, profile_id=1)

        # Test similar content
        final, results = detector.upsert_behavioral_content(
            similar_content, profile_id=2
        )

        reused_sections = [
            section for section, result in results.items() if result.is_similar
        ]

        print(f"   Reused sections: {reused_sections}")
        print(
            f"   Similarity scores: {[f'{s}:{r.similarity_score:.3f}' for s, r in results.items()]}"
        )


def demonstrate_section_wise_analysis():
    """Demonstrate section-wise similarity analysis."""
    print("\n\n=== Section-wise Analysis Demo ===")

    # Create content with mixed similarity levels
    profiles = [
        {
            "about": "Passionate about digital marketing and social media strategy",
            "goalsAndMotivations": ["Build a strong personal brand", "Learn analytics"],
            "frustrations": ["ROI measurement challenges", "Algorithm changes"],
            "needState": "Seeking optimization tools",
            "occasions": "Evening content planning",
        },
        {
            "about": "Enthusiastic about digital marketing and social tactics",  # Similar to #1
            "goalsAndMotivations": [
                "Develop video skills",
                "Master SEO",
            ],  # Different from #1
            "frustrations": [
                "Budget constraints",
                "Time limitations",
            ],  # Different from #1
            "needState": "Looking for automation solutions",  # Different from #1
            "occasions": "Morning and lunch engagement",  # Different from #1
        },
        {
            "about": "Focused on creative content and storytelling",  # Different from #1,#2
            "goalsAndMotivations": [
                "Build a strong personal brand",
                "Learn analytics",
            ],  # Similar to #1
            "frustrations": [
                "ROI measurement challenges",
                "Algorithm changes",
            ],  # Similar to #1
            "needState": "Needing better analytics capabilities",  # Different from #1,#2
            "occasions": "Weekend planning sessions",  # Different from #1,#2
        },
    ]

    detector = SimpleRAGDetector(similarity_threshold=0.7)

    for i, profile in enumerate(profiles, 1):
        print(f"\n--- Adding Profile {i} ---")
        final, results = detector.upsert_behavioral_content(profile, profile_id=i)

        for section, result in results.items():
            status = "REUSED" if result.is_similar else "NEW"
            print(f"   {section}: {status} (Score: {result.similarity_score:.3f})")

    # Final statistics
    print(f"\n--- Final Statistics ---")
    stats = detector.get_stats()
    for section, section_stats in stats["section_stats"].items():
        print(f"   {section}: {section_stats['document_count']} unique items")


if __name__ == "__main__":
    print("🚀 RAG Integration Demo for Synthetic Audience Generator")
    print("=" * 60)

    try:
        # Run demonstrations
        demonstrate_basic_rag_usage()
        demonstrate_enhanced_generator()
        demonstrate_similarity_thresholds()
        demonstrate_section_wise_analysis()

        print("\n" + "=" * 60)
        print("✅ All RAG integration demos completed successfully!")

        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up API keys in .env file")
        print("3. Integrate RAGEnhancedLLMGenerator into your main workflow")
        print("4. Adjust similarity thresholds based on your needs")
        print("5. Monitor RAG statistics to optimize performance")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("\nPlease check your environment setup and try again.")
