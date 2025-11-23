#!/usr/bin/env python3
"""
Simple RAG-based Duplicate Detection System

This module provides a lightweight duplicate detection system using basic
text similarity without heavy dependencies.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class SectionType(str, Enum):
    """Behavioral content section types."""

    ABOUT = "about"
    GOALS = "goalsAndMotivations"
    FRUSTRATIONS = "frustrations"
    NEED_STATE = "needState"
    OCCASIONS = "occasions"


@dataclass
class SimilarityResult:
    """Result of similarity check."""

    is_similar: bool
    similarity_score: float
    similar_content: Optional[str] = None
    section_type: Optional[SectionType] = None


class SimpleTextSimilarity:
    """Simple text similarity calculator using Jaccard similarity."""

    @staticmethod
    def preprocess_text(text: str) -> set:
        """Preprocess text into word tokens."""
        # Convert to lowercase and extract words
        words = re.findall(r"\b\w+\b", text.lower())
        return set(words)

    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        if not text1.strip() or not text2.strip():
            return 0.0

        set1 = SimpleTextSimilarity.preprocess_text(text1)
        set2 = SimpleTextSimilarity.preprocess_text(text2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def list_similarity(list1: List[str], list2: List[str]) -> float:
        """Calculate similarity between two lists of strings."""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0

        # Calculate pairwise similarities and take the maximum
        max_similarities = []

        for item1 in list1:
            item_similarities = [
                SimpleTextSimilarity.jaccard_similarity(item1, item2) for item2 in list2
            ]
            max_similarities.append(
                max(item_similarities) if item_similarities else 0.0
            )

        # Return average of maximum similarities
        return sum(max_similarities) / len(max_similarities)


class SimpleRAGDetector:
    """
    Simple RAG-based duplicate detection system.

    Uses basic text similarity without heavy dependencies for MVP implementation.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize simple RAG detector.

        Args:
            similarity_threshold: Threshold above which content is considered duplicate
        """
        self.similarity_threshold = similarity_threshold

        # Store content by section type
        self.content_store: Dict[SectionType, List[Dict[str, Any]]] = {
            section_type: [] for section_type in SectionType
        }

        # Track document counters
        self.document_counters: Dict[SectionType, int] = {
            section_type: 0 for section_type in SectionType
        }

        logger.info(
            f"Initialized Simple RAG detector with threshold {similarity_threshold}"
        )

    def _prepare_content_for_comparison(
        self, content: Any, section_type: SectionType
    ) -> List[str]:
        """
        Prepare content for comparison based on section type.

        Args:
            content: Raw content (string or list)
            section_type: Type of section

        Returns:
            List of strings ready for comparison
        """
        if section_type in [SectionType.GOALS, SectionType.FRUSTRATIONS]:
            # Handle list content
            if isinstance(content, list):
                return content
            elif isinstance(content, str):
                # Try to parse as JSON list or split by common delimiters
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
                # Split by newlines or other delimiters
                items = [item.strip() for item in content.split("\n") if item.strip()]
                if not items:
                    items = [content]
                return items
            else:
                return [str(content)]
        else:
            # Handle string content (about, needState, occasions)
            return [str(content)]

    def check_similarity(
        self, content: Any, section_type: SectionType
    ) -> SimilarityResult:
        """
        Check if content is similar to existing content in the section.

        Args:
            content: Content to check
            section_type: Section type to check against

        Returns:
            SimilarityResult with similarity information
        """
        stored_content = self.content_store[section_type]

        # If no existing content, it's not similar
        if not stored_content:
            return SimilarityResult(
                is_similar=False, similarity_score=0.0, section_type=section_type
            )

        # Prepare content for comparison
        content_items = self._prepare_content_for_comparison(content, section_type)

        max_similarity = 0.0
        most_similar_content = None

        # Check similarity against all stored content
        for stored_item in stored_content:
            stored_items = stored_item["content_items"]

            if section_type in [SectionType.GOALS, SectionType.FRUSTRATIONS]:
                # Use list similarity for list-based content
                similarity = SimpleTextSimilarity.list_similarity(
                    content_items, stored_items
                )
            else:
                # Use text similarity for string-based content
                if content_items and stored_items:
                    similarity = SimpleTextSimilarity.jaccard_similarity(
                        content_items[0], stored_items[0]
                    )
                else:
                    similarity = 0.0

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_content = stored_items[0] if stored_items else None

        is_similar = max_similarity >= self.similarity_threshold

        return SimilarityResult(
            is_similar=is_similar,
            similarity_score=max_similarity,
            similar_content=most_similar_content,
            section_type=section_type,
        )

    def add_content(
        self,
        content: Any,
        section_type: SectionType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add content to the store if not similar to existing content.

        Args:
            content: Content to add
            section_type: Section type
            metadata: Optional metadata for the document

        Returns:
            True if content was added, False if it was too similar
        """
        # Check similarity first
        similarity_result = self.check_similarity(content, section_type)

        if similarity_result.is_similar:
            logger.info(
                f"Content too similar (score: {similarity_result.similarity_score:.3f}) "
                f"to existing {section_type.value} content. Skipping addition."
            )
            return False

        # Add content to store
        content_items = self._prepare_content_for_comparison(content, section_type)

        self.document_counters[section_type] += 1
        doc_id = f"{section_type.value}_{self.document_counters[section_type]}"

        stored_item = {
            "doc_id": doc_id,
            "content_items": content_items,
            "original_content": content,
            "metadata": metadata or {},
        }

        self.content_store[section_type].append(stored_item)

        logger.info(f"Added content to {section_type.value} store (ID: {doc_id})")
        return True

    def upsert_behavioral_content(
        self, behavioral_content: Dict[str, Any], profile_id: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, SimilarityResult]]:
        """
        Upsert behavioral content, checking similarity section by section.

        Args:
            behavioral_content: Dictionary with behavioral content sections
            profile_id: Optional profile ID for metadata

        Returns:
            Tuple of (final_content, similarity_results)
        """
        final_content = {}
        similarity_results = {}

        metadata = {"profile_id": profile_id} if profile_id else {}

        for section_type in SectionType:
            section_key = section_type.value

            if section_key not in behavioral_content:
                logger.warning(f"Missing section {section_key} in behavioral content")
                continue

            content = behavioral_content[section_key]

            # Check similarity
            similarity_result = self.check_similarity(content, section_type)
            similarity_results[section_key] = similarity_result

            if similarity_result.is_similar:
                logger.info(
                    f"Using existing similar content for {section_key} "
                    f"(similarity: {similarity_result.similarity_score:.3f})"
                )
                # Use the similar content instead of generating new
                final_content[section_key] = similarity_result.similar_content
            else:
                # Use new content and add to store
                final_content[section_key] = content
                self.add_content(content, section_type, metadata)

        return final_content, similarity_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the content stores."""
        stats = {
            "similarity_threshold": self.similarity_threshold,
            "similarity_method": "jaccard",
            "section_stats": {},
        }

        for section_type in SectionType:
            section_content = self.content_store[section_type]
            stats["section_stats"][section_type.value] = {
                "document_count": len(section_content),
                "counter": self.document_counters[section_type],
            }

        return stats

    def clear_section(self, section_type: SectionType) -> bool:
        """
        Clear all content from a specific section.

        Args:
            section_type: Section to clear

        Returns:
            True if successful
        """
        try:
            self.content_store[section_type] = []
            self.document_counters[section_type] = 0
            logger.info(f"Cleared {section_type.value} content store")
            return True
        except Exception as e:
            logger.error(f"Error clearing {section_type.value} content store: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all content stores."""
        try:
            for section_type in SectionType:
                self.clear_section(section_type)
            logger.info("Cleared all content stores")
            return True
        except Exception as e:
            logger.error(f"Error clearing all content stores: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = SimpleRAGDetector(similarity_threshold=0.7)

    # Test behavioral content
    test_content_1 = {
        "about": "Passionate about digital marketing and social media strategy",
        "goalsAndMotivations": [
            "Build a strong personal brand online",
            "Learn advanced analytics tools",
            "Grow professional network",
        ],
        "frustrations": [
            "Difficulty measuring ROI on social campaigns",
            "Keeping up with platform algorithm changes",
        ],
        "needState": "Seeking tools and strategies to optimize social media performance",
        "occasions": "Active during evening hours and weekends for content planning",
    }

    test_content_2 = {
        "about": "Enthusiastic about digital marketing and social media tactics",  # Similar
        "goalsAndMotivations": [
            "Develop expertise in data analysis",  # Different
            "Create engaging video content",
            "Expand client base",
        ],
        "frustrations": [
            "Limited budget for paid advertising",  # Different
            "Time constraints for content creation",
        ],
        "needState": "Looking for cost-effective marketing automation solutions",  # Different
        "occasions": "Most active during morning hours and lunch breaks",  # Different
    }

    # Test upsert functionality
    print("=== Testing Simple RAG Duplicate Detection ===")

    # Add first content
    print("\n1. Adding first behavioral content...")
    final_1, results_1 = detector.upsert_behavioral_content(
        test_content_1, profile_id=1
    )

    for section, result in results_1.items():
        print(
            f"   {section}: Similar={result.is_similar}, Score={result.similarity_score:.3f}"
        )

    # Add second content
    print("\n2. Adding second behavioral content...")
    final_2, results_2 = detector.upsert_behavioral_content(
        test_content_2, profile_id=2
    )

    for section, result in results_2.items():
        status = "REUSED" if result.is_similar else "NEW"
        print(f"   {section}: {status}, Score={result.similarity_score:.3f}")

    # Show stats
    print("\n3. Content store statistics:")
    stats = detector.get_stats()
    print(json.dumps(stats, indent=2))
