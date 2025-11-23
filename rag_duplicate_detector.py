#!/usr/bin/env python3
"""
RAG-based Duplicate Detection System for Synthetic Audience Generator

This module implements section-wise duplicate detection using LangChain's InMemoryVectorStore
to prevent generating similar content across behavioral profile sections.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    # Fallback to basic similarity using TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    EMBEDDINGS_AVAILABLE = False
from pydantic import BaseModel, Field

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


class RAGDuplicateDetector:
    """
    RAG-based duplicate detection system for behavioral content.

    Uses InMemoryVectorStore with section-wise embeddings to detect
    and prevent duplicate content generation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize RAG duplicate detector.

        Args:
            similarity_threshold: Threshold above which content is considered duplicate
            embedding_model: Sentence transformer model for embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

        # Create separate vector stores for each section type
        self.vector_stores: Dict[SectionType, InMemoryVectorStore] = {
            section_type: InMemoryVectorStore(self.embeddings)
            for section_type in SectionType
        }

        # Track document IDs for each section
        self.document_counters: Dict[SectionType, int] = {
            section_type: 0 for section_type in SectionType
        }

        logger.info(
            f"Initialized RAG duplicate detector with threshold {similarity_threshold}"
        )

    def _generate_doc_id(self, section_type: SectionType) -> str:
        """Generate unique document ID for a section."""
        self.document_counters[section_type] += 1
        return f"{section_type.value}_{self.document_counters[section_type]}"

    def _prepare_content_for_embedding(
        self, content: Any, section_type: SectionType
    ) -> List[str]:
        """
        Prepare content for embedding based on section type.

        Args:
            content: Raw content (string or list)
            section_type: Type of section

        Returns:
            List of strings ready for embedding
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
                # Split by common delimiters
                return [item.strip() for item in content.split("\n") if item.strip()]
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
        vector_store = self.vector_stores[section_type]

        # Check if vector store is empty
        try:
            # Try to get all documents to check if store is empty
            all_docs = vector_store.similarity_search("", k=1)
            if not all_docs:
                return SimilarityResult(
                    is_similar=False, similarity_score=0.0, section_type=section_type
                )
        except Exception:
            # Vector store is empty
            return SimilarityResult(
                is_similar=False, similarity_score=0.0, section_type=section_type
            )

        # Prepare content for similarity search
        content_items = self._prepare_content_for_embedding(content, section_type)

        max_similarity = 0.0
        most_similar_content = None

        # Check similarity for each content item
        for item in content_items:
            try:
                # Search for similar documents
                similar_docs = vector_store.similarity_search_with_score(item, k=1)

                if similar_docs:
                    doc, score = similar_docs[0]
                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1.0 - score if score <= 1.0 else 0.0

                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_content = doc.page_content

            except Exception as e:
                logger.warning(f"Error during similarity search: {e}")
                continue

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
        Add content to the vector store if not similar to existing content.

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

        # Add content to vector store
        vector_store = self.vector_stores[section_type]
        content_items = self._prepare_content_for_embedding(content, section_type)

        documents = []
        doc_ids = []

        for item in content_items:
            doc_id = self._generate_doc_id(section_type)
            doc_metadata = {
                "section_type": section_type.value,
                "doc_id": doc_id,
                **(metadata or {}),
            }

            documents.append(Document(page_content=item, metadata=doc_metadata))
            doc_ids.append(doc_id)

        try:
            vector_store.add_documents(documents, ids=doc_ids)
            logger.info(
                f"Added {len(documents)} items to {section_type.value} vector store"
            )
            return True
        except Exception as e:
            logger.error(f"Error adding content to vector store: {e}")
            return False

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
                # Use new content and add to vector store
                final_content[section_key] = content
                self.add_content(content, section_type, metadata)

        return final_content, similarity_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector stores."""
        stats = {}

        for section_type in SectionType:
            try:
                # Get all documents to count them
                all_docs = self.vector_stores[section_type].similarity_search(
                    "", k=1000
                )
                doc_count = len(all_docs)
            except Exception:
                doc_count = 0

            stats[section_type.value] = {
                "document_count": doc_count,
                "counter": self.document_counters[section_type],
            }

        return {
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embeddings.model_name,
            "section_stats": stats,
        }

    def clear_section(self, section_type: SectionType) -> bool:
        """
        Clear all documents from a specific section.

        Args:
            section_type: Section to clear

        Returns:
            True if successful
        """
        try:
            # Reinitialize the vector store for this section
            self.vector_stores[section_type] = InMemoryVectorStore(self.embeddings)
            self.document_counters[section_type] = 0
            logger.info(f"Cleared {section_type.value} vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing {section_type.value} vector store: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all vector stores."""
        try:
            for section_type in SectionType:
                self.clear_section(section_type)
            logger.info("Cleared all vector stores")
            return True
        except Exception as e:
            logger.error(f"Error clearing all vector stores: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = RAGDuplicateDetector(similarity_threshold=0.85)

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
        "about": "Enthusiastic about digital marketing and social media tactics",  # Similar to test_content_1
        "goalsAndMotivations": [
            "Develop expertise in data analysis",  # Different from test_content_1
            "Create engaging video content",
            "Expand client base",
        ],
        "frustrations": [
            "Limited budget for paid advertising",  # Different from test_content_1
            "Time constraints for content creation",
        ],
        "needState": "Looking for cost-effective marketing automation solutions",  # Different
        "occasions": "Most active during morning hours and lunch breaks",  # Different
    }

    # Test upsert functionality
    print("=== Testing RAG Duplicate Detection ===")

    # Add first content
    print("\n1. Adding first behavioral content...")
    final_1, results_1 = detector.upsert_behavioral_content(
        test_content_1, profile_id=1
    )

    for section, result in results_1.items():
        print(
            f"   {section}: Similar={result.is_similar}, Score={result.similarity_score:.3f}"
        )

    # Add second content (should detect similarities in 'about' section)
    print("\n2. Adding second behavioral content...")
    final_2, results_2 = detector.upsert_behavioral_content(
        test_content_2, profile_id=2
    )

    for section, result in results_2.items():
        print(
            f"   {section}: Similar={result.is_similar}, Score={result.similarity_score:.3f}"
        )

    # Show stats
    print("\n3. Vector store statistics:")
    stats = detector.get_stats()
    print(json.dumps(stats, indent=2))
