#!/usr/bin/env python3
"""
Example Usage of Synthetic Audience Generator

This script demonstrates how to use the main synthetic_audience_mvp.py
"""

import subprocess
import json
import os
from pathlib import Path


def run_generation_example():
    """Run the synthetic audience generator with example data."""

    print("🎯 Synthetic Audience Generator - Example Usage")
    print("=" * 60)

    # Check if API key is configured
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment")
        print("💡 Please configure your .env file:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Google API key to .env")
        return

    # Check if input file exists
    input_file = "dataset/small_demo_input.json"
    if not Path(input_file).exists():
        print(f"❌ Error: Input file {input_file} not found")
        return

    # Run the generation
    output_file = "results/example_output.json"

    print(f"📥 Input: {input_file}")
    print(f"📤 Output: {output_file}")
    print("\n🚀 Starting generation...")

    try:
        # Run the main script
        result = subprocess.run(
            [
                "python",
                "synthetic_audience_mvp.py",
                "-i",
                input_file,
                "-o",
                output_file,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("✅ Generation completed successfully!")

            # Show results summary
            if Path(output_file).exists():
                with open(output_file, "r") as f:
                    data = json.load(f)

                profiles = data.get("synthetic_audience", [])
                metadata = data.get("generation_metadata", {})

                print(f"\n📊 Results Summary:")
                print(f"   Total Profiles: {len(profiles)}")

                if metadata.get("distribution_accuracy"):
                    dist = metadata["distribution_accuracy"]
                    print(
                        f"   Gender Distribution: {dist.get('gender_distribution', {}).get('actual_counts', {})}"
                    )
                    print(
                        f"   Age Distribution: {dist.get('age_distribution', {}).get('actual_counts', {})}"
                    )

                print(f"\n📄 Output saved to: {output_file}")
                print("\n🎉 Example completed successfully!")

        else:
            print("❌ Generation failed!")
            print(f"Error: {result.stderr}")

            if "quota" in result.stderr.lower():
                print("\n💡 Quota exceeded. Try:")
                print("   1. Wait 24 hours for quota reset")
                print("   2. Switch to gemini-1.5-flash in .env")
                print("   3. Upgrade to paid API tier")

    except subprocess.TimeoutExpired:
        print("⏰ Generation timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Error running generation: {e}")


def show_input_format():
    """Show the expected input format."""

    print("\n📋 Expected Input Format:")
    print("-" * 30)

    example_input = {
        "request": [
            {
                "filter_details": {
                    "user_req_responses": 5,
                    "gender_proportions": {"Female": 80, "Male": 20},
                    "age_proportions": {
                        "GenZ": 40,
                        "Young Millennials": 40,
                        "Old Millennials": 10,
                        "GenX": 5,
                        "Boomer": 5,
                    },
                    "ethnicity_proportions": {
                        "White/Caucasian": 59,
                        "Asian": 6,
                        "Black or African-American": 15,
                        "Native Hawaiian or Other Pacific Islander": 1,
                        "Hispanics": 19,
                    },
                },
                "personas": [
                    {
                        "about": "Template persona description...",
                        "goalsAndMotivations": ["Goal 1", "Goal 2"],
                        "frustrations": ["Frustration 1", "Frustration 2"],
                    }
                ],
            }
        ]
    }

    print(json.dumps(example_input, indent=2))


def main():
    """Main example function."""

    print("Choose an option:")
    print("1. Run generation example")
    print("2. Show input format")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        run_generation_example()
    elif choice == "2":
        show_input_format()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
