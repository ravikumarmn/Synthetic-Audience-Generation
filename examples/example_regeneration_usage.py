#!/usr/bin/env python3
"""
Example Usage of RAG Regeneration Workflow

This script demonstrates how to use the RAG workflow that:
1. Generates audience profiles in parallel
2. Detects duplicates using RAG
3. Regenerates content when duplicates are found
4. Writes only unique profiles to JSON
"""

import json
import asyncio
import logging
from typing import Dict, Any

from rag_regeneration_workflow import create_rag_regeneration_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_input(filename: str = "sample_input.json", num_profiles: int = 10):
    """Create a sample input file for testing."""
    sample_data = {
        "filterDetails": {
            "user_req_responses": num_profiles,
            "age_filter": {"min": 25, "max": 45},
            "age_proportions": {"25-34": 60, "35-44": 40},
            "gender_filter": ["Male", "Female"],
            "gender_proportions": {"Male": 50, "Female": 50},
            "ethnicity_filter": ["White", "Hispanic", "Black"],
            "ethnicity_proportions": {"White": 60, "Hispanic": 25, "Black": 15},
        },
        "personas": [
            {
                "id": 1,
                "generatedTitle": "Digital Marketing Professional",
                "about": "Passionate about digital marketing and social media strategy with focus on ROI optimization",
                "goalsAndMotivations": [
                    "Build a strong personal brand online",
                    "Learn advanced analytics and data visualization tools",
                    "Grow professional network in marketing industry",
                ],
                "frustrations": [
                    "Difficulty measuring ROI on social media campaigns",
                    "Keeping up with constantly changing platform algorithms",
                    "Limited budget for paid advertising experiments",
                ],
                "needState": "Seeking comprehensive tools and strategies to optimize social media performance and demonstrate clear business value",
                "occasions": "Most active during evening hours and weekends for content planning and strategy development",
            },
            {
                "id": 2,
                "generatedTitle": "Content Creator Entrepreneur",
                "about": "Creative professional focused on authentic storytelling and community building through various digital platforms",
                "goalsAndMotivations": [
                    "Create viral content that resonates with target audience",
                    "Monetize creative skills through multiple revenue streams",
                    "Build an engaged community of loyal followers",
                ],
                "frustrations": [
                    "Inconsistent content performance across platforms",
                    "Time management between content creation and business development",
                    "Finding authentic brand partnerships that align with values",
                ],
                "needState": "Looking for streamlined content creation workflows and authentic monetization opportunities",
                "occasions": "Peak creativity during morning hours and late evening sessions, with weekend batch content creation",
            },
            {
                "id": 3,
                "generatedTitle": "Small Business Owner",
                "about": "Entrepreneur running a local service business with growing online presence and customer base",
                "goalsAndMotivations": [
                    "Expand customer base through digital marketing",
                    "Improve operational efficiency and customer service",
                    "Build sustainable business growth strategies",
                ],
                "frustrations": [
                    "Limited time for marketing while running daily operations",
                    "Difficulty competing with larger businesses online",
                    "Managing customer expectations and service quality",
                ],
                "needState": "Needs cost-effective marketing automation and customer management solutions",
                "occasions": "Business planning during early morning hours and customer engagement throughout business hours",
            },
        ],
    }

    with open(filename, "w") as f:
        json.dump(sample_data, f, indent=2)

    logger.info(f"✅ Created sample input file: {filename}")
    return filename


async def run_regeneration_workflow_example():
    """Run the RAG regeneration workflow example."""
    print("🚀 RAG Regeneration Workflow Example")
    print("=" * 50)

    # Configuration
    config = {
        "similarity_threshold": 0.75,  # Lower threshold to trigger more regenerations
        "max_regeneration_attempts": 3,
        "batch_size": 3,
        "num_profiles": 8,
    }

    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    # Create sample input
    input_file = create_sample_input("sample_input.json", config["num_profiles"])
    output_file = "regeneration_output.json"

    # Create workflow
    workflow = create_rag_regeneration_workflow(
        similarity_threshold=config["similarity_threshold"],
        max_regeneration_attempts=config["max_regeneration_attempts"],
        batch_size=config["batch_size"],
    )

    app = workflow.compile()

    # Initial state
    initial_state = {
        "input_file": input_file,
        "output_file": output_file,
        "similarity_threshold": config["similarity_threshold"],
        "max_regeneration_attempts": config["max_regeneration_attempts"],
        "batch_size": config["batch_size"],
    }

    print(f"\n🔄 Running workflow...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")

    try:
        # Run the workflow (use async invoke for async nodes)
        final_state = await app.ainvoke(initial_state)

        print(f"\n✅ Workflow completed successfully!")

        # Display results
        generation_stats = final_state.get("generation_stats", {})
        print(f"\n📊 Generation Statistics:")
        print(f"  - Total requested: {generation_stats.get('total_requested', 0)}")
        print(
            f"  - Total generated (unique): {generation_stats.get('total_generated', 0)}"
        )
        print(
            f"  - Total duplicates rejected: {generation_stats.get('total_duplicates_rejected', 0)}"
        )

        regeneration_stats = generation_stats.get("regeneration_stats", {})
        if regeneration_stats:
            print(f"\n� Regeneration Statistics:")
            print(
                f"  - Total regenerations attempted: {regeneration_stats.get('total_regenerations', 0)}"
            )
            print(
                f"  - Successful regenerations: {regeneration_stats.get('successful_regenerations', 0)}"
            )
            print(
                f"  - Failed regenerations: {regeneration_stats.get('failed_regenerations', 0)}"
            )

        rag_stats = generation_stats.get("rag_stats", {})
        if rag_stats:
            print(f"\n🧠 RAG System Statistics:")
            print(
                f"  - Similarity threshold: {rag_stats.get('similarity_threshold', 'N/A')}"
            )
            section_stats = rag_stats.get("section_stats", {})
            for section, stats in section_stats.items():
                print(f"  - {section}: {stats.get('document_count', 0)} unique items")

        # Show sample output
        if final_state.get("synthetic_profiles"):
            print(f"\n📄 Sample Generated Profile:")
            sample_profile = final_state["synthetic_profiles"][0]
            print(f"  Profile ID: {sample_profile.get('profile_id', 'N/A')}")
            print(f"  About: {sample_profile.get('about', 'N/A')[:80]}...")
            print(
                f"  Goals: {len(sample_profile.get('goalsAndMotivations', []))} items"
            )
            print(
                f"  Frustrations: {len(sample_profile.get('frustrations', []))} items"
            )

        return True

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        logger.error(f"Workflow execution failed: {str(e)}")
        return False


def demonstrate_regeneration_logic():
    """Demonstrate the regeneration logic conceptually."""
    print("\n🔄 Regeneration Logic Demonstration")
    print("=" * 40)

    print("1. 📝 Generate Profile Content")
    print("   - LLM creates behavioral content for profile")
    print("   - Content includes: about, goals, frustrations, needState, occasions")

    print("\n2. 🔍 RAG Duplicate Detection")
    print("   - Check each section against existing content")
    print("   - Calculate similarity scores using Jaccard similarity")
    print("   - Compare against configurable thresholds")

    print("\n3. 🎯 Decision Logic")
    print("   - If NO duplicates found → ✅ Add to unique profiles")
    print("   - If duplicates found → 🔄 Regenerate with variations")
    print("   - If max attempts reached → ❌ Mark as duplicate")

    print("\n4. 💾 Output Management")
    print("   - Write ONLY unique profiles to JSON file")
    print("   - Track duplicate profiles separately")
    print("   - Generate comprehensive statistics")

    print("\n5. 📊 Benefits")
    print("   - Ensures content uniqueness")
    print("   - Prevents duplicate audience profiles")
    print("   - Maintains high content quality")
    print("   - Provides detailed analytics")


async def main():
    """Main function."""
    print("🎨 RAG Regeneration Workflow - Complete Example")
    print("=" * 60)

    # Demonstrate the logic
    demonstrate_regeneration_logic()

    # Run the actual workflow
    success = await run_regeneration_workflow_example()

    if success:
        print(f"\n🎉 Example completed successfully!")
        print(f"\n📁 Files created:")
        print(f"  - sample_input.json (input data)")
        print(f"  - regeneration_output.json (unique profiles only)")
        print(f"  - workflow_rag_regeneration.png (workflow diagram)")

        print(f"\n🔧 Next Steps:")
        print(f"  1. Adjust similarity thresholds based on your needs")
        print(f"  2. Modify regeneration attempts for your use case")
        print(f"  3. Integrate with your actual LLM provider")
        print(f"  4. Scale batch size for better performance")
        print(f"  5. Monitor regeneration statistics")

    else:
        print(f"\n❌ Example failed. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
