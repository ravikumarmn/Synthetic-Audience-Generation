#!/usr/bin/env python3
"""
Cosine Similarity Analysis Script
Calculates similarity between input personas and synthetic audience profiles.
Score: 0 = identical, 1 = completely different
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import os
import warnings

warnings.filterwarnings("ignore")


class CosineSimilarityAnalyzer:
    """Comprehensive cosine similarity analyzer for persona comparison"""

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.input_data = {}
        self.output_data = {}
        self.input_texts = []
        self.synthetic_texts = []

    def load_data(self) -> bool:
        """Load input and output JSON files"""
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                self.input_data = json.load(f)
            print(f"✅ Input data loaded: {self.input_file}")

            with open(self.output_file, "r", encoding="utf-8") as f:
                self.output_data = json.load(f)
            print(f"✅ Output data loaded: {self.output_file}")

            return True
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return False

    def extract_persona_text(self, persona: Dict[str, Any]) -> Dict[str, str]:
        """Extract text content from persona for similarity analysis"""
        text_fields = {
            "about": persona.get("about", ""),
            "goals": " ".join(persona.get("goalsAndMotivations", [])),
            "frustrations": " ".join(persona.get("frustrations", [])),
            "need_state": persona.get("needState", ""),
            "occasions": persona.get("occasions", ""),
        }

        # Create combined text for overall similarity
        text_fields["combined"] = " ".join(
            [
                text_fields["about"],
                text_fields["goals"],
                text_fields["frustrations"],
                text_fields["need_state"],
                text_fields["occasions"],
            ]
        ).strip()

        return text_fields

    def extract_synthetic_text(self, profile: Dict[str, Any]) -> Dict[str, str]:
        """Extract text content from synthetic profile"""
        behavioral_content = profile.get("behavioral_content", {})

        text_fields = {
            "about": behavioral_content.get("about", ""),
            "goals": " ".join(behavioral_content.get("goals", [])),
            "frustrations": " ".join(behavioral_content.get("frustrations", [])),
            "need_state": behavioral_content.get("needState", ""),
            "occasions": behavioral_content.get("occasions", ""),
        }

        # Create combined text for overall similarity
        text_fields["combined"] = " ".join(
            [
                text_fields["about"],
                text_fields["goals"],
                text_fields["frustrations"],
                text_fields["need_state"],
                text_fields["occasions"],
            ]
        ).strip()

        return text_fields

    def extract_all_texts(self):
        """Extract text from all personas and profiles"""
        # Extract from input personas
        input_personas = self.input_data.get("request", [{}])[0].get("personas", [])
        self.input_texts = [
            self.extract_persona_text(persona) for persona in input_personas
        ]

        # Extract from synthetic profiles
        synthetic_profiles = self.output_data.get("synthetic_audience", [])
        self.synthetic_texts = [
            self.extract_synthetic_text(profile) for profile in synthetic_profiles
        ]

        print(f"📝 Extracted text from {len(self.input_texts)} input personas")
        print(f"📝 Extracted text from {len(self.synthetic_texts)} synthetic profiles")

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        if not text1.strip() or not text2.strip():
            return 1.0  # Completely different if either is empty

        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0, 1]

            # Convert to dissimilarity (0 = identical, 1 = completely different)
            dissimilarity_score = 1 - similarity_score

            return round(dissimilarity_score, 4)

        except Exception as e:
            print(f"⚠️ Error calculating similarity: {e}")
            return 1.0

    def get_similarity_interpretation(self, score: float) -> str:
        """Interpret similarity score"""
        if score <= 0.2:
            return "Very Similar"
        elif score <= 0.4:
            return "Similar"
        elif score <= 0.6:
            return "Moderately Different"
        elif score <= 0.8:
            return "Different"
        else:
            return "Very Different"

    def analyze_similarity(self) -> pd.DataFrame:
        """Perform comprehensive similarity analysis"""
        results = []

        if not self.synthetic_texts:
            print(
                "⚠️ No synthetic profiles found. Creating demo analysis with input personas only."
            )

            # Self-comparison for demonstration
            for i, input_text in enumerate(self.input_texts):
                for j, compare_text in enumerate(self.input_texts):
                    if i != j:  # Don't compare with itself
                        for section in [
                            "about",
                            "goals",
                            "frustrations",
                            "need_state",
                            "occasions",
                            "combined",
                        ]:
                            score = self.calculate_cosine_similarity(
                                input_text[section], compare_text[section]
                            )

                            results.append(
                                {
                                    "input_persona_id": i + 1,
                                    "synthetic_profile_id": f"Input_{j + 1}",
                                    "section": section,
                                    "similarity_score": score,
                                    "interpretation": self.get_similarity_interpretation(
                                        score
                                    ),
                                    "input_text_length": len(input_text[section]),
                                    "synthetic_text_length": len(compare_text[section]),
                                    "comparison_type": "Input-to-Input (Demo)",
                                }
                            )
        else:
            # Real comparison between input and synthetic
            for i, input_text in enumerate(self.input_texts):
                for j, synthetic_text in enumerate(self.synthetic_texts):
                    for section in [
                        "about",
                        "goals",
                        "frustrations",
                        "need_state",
                        "occasions",
                        "combined",
                    ]:
                        score = self.calculate_cosine_similarity(
                            input_text[section], synthetic_text[section]
                        )

                        results.append(
                            {
                                "input_persona_id": i + 1,
                                "synthetic_profile_id": j + 1,
                                "section": section,
                                "similarity_score": score,
                                "interpretation": self.get_similarity_interpretation(
                                    score
                                ),
                                "input_text_length": len(input_text[section]),
                                "synthetic_text_length": len(synthetic_text[section]),
                                "comparison_type": "Input-to-Synthetic",
                            }
                        )

        return pd.DataFrame(results)

    def generate_reports(self, similarity_df: pd.DataFrame):
        """Generate comprehensive analysis reports"""
        print("\n🎯 SIMILARITY ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Total Comparisons: {len(similarity_df)}")
        print(
            f"Average Similarity Score: {similarity_df['similarity_score'].mean():.4f}"
        )
        print(
            f"Score Range: {similarity_df['similarity_score'].min():.4f} - {similarity_df['similarity_score'].max():.4f}"
        )

        print("\n📋 Score Distribution by Interpretation:")
        print(similarity_df["interpretation"].value_counts())

        # Section-wise analysis
        section_analysis = (
            similarity_df.groupby("section")
            .agg(
                {
                    "similarity_score": ["mean", "std", "min", "max", "count"],
                    "interpretation": lambda x: (
                        x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"
                    ),
                }
            )
            .round(4)
        )

        section_analysis.columns = [
            "Mean_Score",
            "Std_Dev",
            "Min_Score",
            "Max_Score",
            "Count",
            "Most_Common_Interpretation",
        ]
        section_analysis = section_analysis.sort_values("Mean_Score")

        print("\n📊 SECTION-WISE SIMILARITY ANALYSIS")
        print("=" * 60)
        print(section_analysis)

        # Best matches
        if "combined" in similarity_df["section"].values:
            combined_data = similarity_df[similarity_df["section"] == "combined"].copy()
            best_matches = combined_data.loc[
                combined_data.groupby("input_persona_id")["similarity_score"].idxmin()
            ]

            print("\n🏆 BEST MATCHES (Lowest Dissimilarity Scores)")
            print("=" * 55)
            print(
                best_matches[
                    [
                        "input_persona_id",
                        "synthetic_profile_id",
                        "similarity_score",
                        "interpretation",
                    ]
                ]
            )

        # Summary
        print("\n📋 FINAL SUMMARY")
        print("=" * 30)
        print(
            f"Most Similar Section: {section_analysis.index[0]} (Score: {section_analysis.iloc[0]['Mean_Score']:.4f})"
        )
        print(
            f"Most Different Section: {section_analysis.index[-1]} (Score: {section_analysis.iloc[-1]['Mean_Score']:.4f})"
        )
        print(
            f"Overall Average Dissimilarity: {similarity_df['similarity_score'].mean():.4f}"
        )
        print(
            f"🎯 Interpretation: {self.get_similarity_interpretation(similarity_df['similarity_score'].mean())}"
        )

        return section_analysis, (
            best_matches if "combined" in similarity_df["section"].values else None
        )

    def save_results(self, similarity_df: pd.DataFrame, section_analysis: pd.DataFrame):
        """Save results to CSV files"""
        # Create results directory if it doesn't exist
        os.makedirs("../results", exist_ok=True)

        # Save detailed results
        output_filename = "../results/cosine_similarity_analysis.csv"
        similarity_df.to_csv(output_filename, index=False)
        print(f"\n💾 Detailed results exported to: {output_filename}")

        # Save section analysis
        section_filename = "../results/section_wise_similarity.csv"
        section_analysis.to_csv(section_filename)
        print(f"💾 Section analysis exported to: {section_filename}")

    def run_analysis(self):
        """Run complete similarity analysis"""
        print("📊 Starting Cosine Similarity Analysis")
        print("=" * 50)

        # Load data
        if not self.load_data():
            return

        # Extract texts
        self.extract_all_texts()

        # Perform analysis
        similarity_df = self.analyze_similarity()
        print(f"\n📊 Analysis complete: {len(similarity_df)} comparisons")

        # Generate reports
        section_analysis, best_matches = self.generate_reports(similarity_df)

        # Save results
        self.save_results(similarity_df, section_analysis)

        return similarity_df, section_analysis, best_matches


def main():
    """Main execution function"""
    # File paths
    input_file = "./dataset/small_demo_input.json"
    output_file = "./results/output.json"

    # Create analyzer and run analysis
    analyzer = CosineSimilarityAnalyzer(input_file, output_file)
    results = analyzer.run_analysis()

    if results:
        similarity_df, section_analysis, best_matches = results
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 {len(similarity_df)} total comparisons analyzed")
        print(f"📁 Results saved to ../results/ directory")


if __name__ == "__main__":
    main()
