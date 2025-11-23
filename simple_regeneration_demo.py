#!/usr/bin/env python3
"""
Simple RAG Regeneration Demo

This demonstrates the core regeneration logic without complex async workflows.
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from simple_rag_detector import SimpleRAGDetector, SimilarityResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRegenerationEngine:
    """Simple engine that demonstrates regeneration logic."""

    def __init__(self, similarity_threshold: float = 0.8, max_attempts: int = 3):
        self.rag_detector = SimpleRAGDetector(similarity_threshold=similarity_threshold)
        self.max_attempts = max_attempts
        self.stats = {
            "total_generated": 0,
            "total_duplicates": 0,
            "total_regenerations": 0,
            "successful_regenerations": 0,
        }

    def generate_profile_content(
        self, profile_id: int, attempt: int = 1
    ) -> Dict[str, Any]:
        """Simulate LLM content generation with variations per attempt."""
        base_content = {
            "about": f"Marketing professional with {attempt} years of digital strategy experience (Profile {profile_id})",
            "goalsAndMotivations": [
                f"Build personal brand online (attempt {attempt})",
                f"Learn analytics tools (variation {attempt})",
                f"Grow network in marketing (try {attempt})",
            ],
            "frustrations": [
                f"ROI measurement challenges (attempt {attempt})",
                f"Algorithm changes (variation {attempt})",
            ],
            "needState": f"Seeking optimization tools and strategies (attempt {attempt})",
            "occasions": f"Active during evening hours for planning (attempt {attempt})",
        }

        # Add more variation for higher attempts
        if attempt > 1:
            base_content["about"] += f" with specialized focus area {attempt}"
            base_content["goalsAndMotivations"].append(f"Additional goal {attempt}")
            base_content["frustrations"].append(f"New challenge {attempt}")

        return base_content

    def generate_with_regeneration(
        self, profile_id: int
    ) -> Tuple[Dict[str, Any], bool, int]:
        """
        Generate profile with regeneration logic.

        Returns:
            (final_content, was_duplicate, attempts_used)
        """
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"🔄 Generating profile {profile_id}, attempt {attempt}")

            # Generate content
            content = self.generate_profile_content(profile_id, attempt)

            # Check for duplicates
            final_content, similarity_results = (
                self.rag_detector.upsert_behavioral_content(
                    content, profile_id=profile_id
                )
            )

            # Check if any section was similar (duplicate)
            has_duplicates = any(
                result.is_similar for result in similarity_results.values()
            )

            if not has_duplicates:
                # Success - no duplicates
                logger.info(
                    f"✅ Profile {profile_id} generated successfully (attempt {attempt})"
                )
                self.stats["total_generated"] += 1
                if attempt > 1:
                    self.stats["successful_regenerations"] += 1
                self.stats["total_regenerations"] += attempt - 1
                return final_content, False, attempt
            else:
                # Duplicates found
                duplicate_sections = [
                    section
                    for section, result in similarity_results.items()
                    if result.is_similar
                ]
                logger.info(
                    f"🔄 Profile {profile_id} has duplicates in {duplicate_sections}"
                )

                if attempt == self.max_attempts:
                    # Max attempts reached
                    logger.warning(
                        f"❌ Profile {profile_id} still has duplicates after {attempt} attempts"
                    )
                    self.stats["total_duplicates"] += 1
                    self.stats["total_regenerations"] += attempt
                    return final_content, True, attempt

        # Should not reach here
        return {}, True, self.max_attempts


def demonstrate_regeneration():
    """Demonstrate the regeneration logic."""
    print("🚀 Simple RAG Regeneration Demo")
    print("=" * 40)

    # Create engine with lower threshold to trigger more regenerations
    engine = SimpleRegenerationEngine(similarity_threshold=0.7, max_attempts=3)

    print(f"📋 Configuration:")
    print(f"  - Similarity threshold: 0.7")
    print(f"  - Max regeneration attempts: 3")

    # Generate multiple profiles to see regeneration in action
    profiles = []
    duplicate_profiles = []

    print(f"\n🔄 Generating 8 profiles...")

    for profile_id in range(1, 9):
        print(f"\n--- Profile {profile_id} ---")

        final_content, was_duplicate, attempts = engine.generate_with_regeneration(
            profile_id
        )

        if was_duplicate:
            duplicate_profiles.append(
                {
                    "profile_id": profile_id,
                    "content": final_content,
                    "attempts": attempts,
                }
            )
            print(f"❌ Profile {profile_id}: DUPLICATE (after {attempts} attempts)")
        else:
            profiles.append(
                {
                    "profile_id": profile_id,
                    "content": final_content,
                    "attempts": attempts,
                }
            )
            print(f"✅ Profile {profile_id}: UNIQUE (after {attempts} attempts)")

    # Show results
    print(f"\n📊 Final Results:")
    print(f"  - Unique profiles: {len(profiles)}")
    print(f"  - Duplicate profiles: {len(duplicate_profiles)}")
    print(f"  - Total regenerations: {engine.stats['total_regenerations']}")
    print(f"  - Successful regenerations: {engine.stats['successful_regenerations']}")

    # Write unique profiles to JSON
    output_data = {
        "unique_profiles": profiles,
        "duplicate_profiles": duplicate_profiles,
        "statistics": engine.stats,
        "rag_stats": engine.rag_detector.get_stats(),
    }

    with open("simple_regeneration_output.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: simple_regeneration_output.json")

    # Show sample content
    if profiles:
        print(f"\n📄 Sample Unique Profile:")
        sample = profiles[0]
        content = sample["content"]
        print(f"  Profile ID: {sample['profile_id']}")
        print(f"  Attempts: {sample['attempts']}")
        print(f"  About: {content.get('about', 'N/A')[:60]}...")
        print(f"  Goals: {len(content.get('goalsAndMotivations', []))} items")

    # Show RAG statistics
    rag_stats = engine.rag_detector.get_stats()
    print(f"\n🧠 RAG Statistics:")
    print(f"  - Similarity threshold: {rag_stats['similarity_threshold']}")
    section_stats = rag_stats.get("section_stats", {})
    for section, stats in section_stats.items():
        print(f"  - {section}: {stats['document_count']} unique items")

    return len(profiles), len(duplicate_profiles)


def explain_workflow():
    """Explain the regeneration workflow."""
    print("\n📋 Regeneration Workflow Explanation:")
    print("=" * 45)

    print("1. 🎯 Profile Generation Request")
    print("   → LLM generates behavioral content")
    print("   → Content includes all required sections")

    print("\n2. 🔍 RAG Duplicate Detection")
    print("   → Check each section against existing content")
    print("   → Calculate similarity scores")
    print("   → Compare against threshold")

    print("\n3. 🔄 Decision Logic")
    print("   → If NO duplicates: ✅ Accept profile")
    print("   → If duplicates found: 🔄 Regenerate with variations")
    print("   → If max attempts: ❌ Mark as duplicate")

    print("\n4. 💾 Output Management")
    print("   → Write ONLY unique profiles to JSON")
    print("   → Track duplicates separately")
    print("   → Generate comprehensive statistics")

    print("\n5. 🎯 Benefits")
    print("   → Ensures content uniqueness")
    print("   → Prevents duplicate audience profiles")
    print("   → Maintains quality standards")
    print("   → Provides detailed analytics")


def main():
    """Main demo function."""
    explain_workflow()

    unique_count, duplicate_count = demonstrate_regeneration()

    print(f"\n🎉 Demo completed!")
    print(f"📊 Summary: {unique_count} unique, {duplicate_count} duplicates")

    if unique_count > 0:
        print(f"✅ Regeneration logic working correctly!")
        print(f"📁 Check 'simple_regeneration_output.json' for detailed results")
    else:
        print(
            f"⚠️  All profiles were duplicates - consider lowering similarity threshold"
        )

    print(f"\n🔧 Next Steps:")
    print(f"  1. Integrate with your actual LLM provider")
    print(f"  2. Adjust similarity thresholds based on your needs")
    print(f"  3. Implement parallel processing for better performance")
    print(f"  4. Add more sophisticated content variation strategies")


if __name__ == "__main__":
    main()
