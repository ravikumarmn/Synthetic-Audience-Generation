#!/usr/bin/env python3
"""
RAG-Enhanced Content Generator for Synthetic Audience Generator

This module extends the existing LLMContentGenerator with RAG-based duplicate detection
to prevent generating similar behavioral content.
"""

import json
import os
import re
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

# Import existing models and RAG detector
from rag_duplicate_detector import RAGDuplicateDetector, SimilarityResult
from prompt import BEHAVIORAL_CONTENT_PROMPT

logger = logging.getLogger(__name__)


class BehavioralContent(BaseModel):
    """Generated behavioral content from LLM."""

    about: str
    goalsAndMotivations: List[str]
    frustrations: List[str]
    needState: str
    occasions: str


class ProcessingTemplates(BaseModel):
    """Cleaned templates for LLM generation."""

    about_templates: List[str]
    goals_templates: List[str]
    frustrations_templates: List[str]
    need_state_templates: List[str]
    occasions_templates: List[str]


@dataclass
class RAGGenerationResult:
    """Result of RAG-enhanced content generation."""

    content: BehavioralContent
    similarity_results: Dict[str, SimilarityResult]
    was_generated: (
        bool  # True if new content was generated, False if existing was reused
    )
    generation_attempts: int


class RAGEnhancedLLMGenerator:
    """
    LLM Content Generator enhanced with RAG-based duplicate detection.

    This class extends the original LLMContentGenerator to include:
    - Section-wise duplicate detection
    - Intelligent content reuse
    - Similarity-based content filtering
    """

    def __init__(
        self,
        provider: str = None,
        similarity_threshold: float = 0.85,
        enable_rag: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize RAG-enhanced LLM generator.

        Args:
            provider: LLM provider ("google" or "azure")
            similarity_threshold: Threshold for duplicate detection
            enable_rag: Whether to enable RAG duplicate detection
            embedding_model: Sentence transformer model for embeddings
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "google")
        self.enable_rag = enable_rag

        # Initialize RAG duplicate detector
        if self.enable_rag:
            self.rag_detector = RAGDuplicateDetector(
                similarity_threshold=similarity_threshold,
                embedding_model=embedding_model,
            )
            logger.info(
                f"RAG duplicate detection enabled with threshold {similarity_threshold}"
            )
        else:
            self.rag_detector = None
            logger.info("RAG duplicate detection disabled")

        # Initialize LLM
        self._initialize_llm()

        # Initialize prompt template
        self.prompt_template = PromptTemplate(
            input_variables=[
                "about_examples",
                "goals_examples",
                "frustrations_examples",
                "need_state_examples",
                "occasions_examples",
            ],
            template=BEHAVIORAL_CONTENT_PROMPT,
        )

        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider."""
        if self.provider == "azure":
            # Azure OpenAI configuration
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

            if not all([azure_endpoint, api_key, deployment_name]):
                raise ValueError(
                    "Azure OpenAI configuration missing. Required: "
                    "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME"
                )

            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                azure_deployment=deployment_name,
                api_version=api_version,
                temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "1500")),
            )
        else:
            # Google Gemini configuration (default)
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1500")),
                google_api_key=api_key,
            )

    def generate_content_with_rag(
        self,
        templates: ProcessingTemplates,
        profile_id: Optional[int] = None,
        force_generation: bool = False,
    ) -> RAGGenerationResult:
        """
        Generate behavioral content with RAG-based duplicate detection.

        Args:
            templates: Processing templates for content generation
            profile_id: Optional profile ID for metadata
            force_generation: If True, skip RAG check and always generate new content

        Returns:
            RAGGenerationResult with content and similarity information
        """
        if not self.enable_rag or force_generation:
            # Generate without RAG
            content = self._generate_content_direct(templates)
            return RAGGenerationResult(
                content=content,
                similarity_results={},
                was_generated=True,
                generation_attempts=1,
            )

        # Generate new content first
        attempts = 0
        max_generation_attempts = 3

        while attempts < max_generation_attempts:
            attempts += 1

            try:
                # Generate new behavioral content
                new_content = self._generate_content_direct(templates)

                # Convert to dictionary for RAG processing
                content_dict = {
                    "about": new_content.about,
                    "goalsAndMotivations": new_content.goalsAndMotivations,
                    "frustrations": new_content.frustrations,
                    "needState": new_content.needState,
                    "occasions": new_content.occasions,
                }

                # Check similarity and upsert with RAG
                final_content_dict, similarity_results = (
                    self.rag_detector.upsert_behavioral_content(
                        content_dict, profile_id=profile_id
                    )
                )

                # Convert back to BehavioralContent
                final_content = BehavioralContent(**final_content_dict)

                # Determine if any content was reused
                was_generated = not any(
                    result.is_similar for result in similarity_results.values()
                )

                return RAGGenerationResult(
                    content=final_content,
                    similarity_results=similarity_results,
                    was_generated=was_generated,
                    generation_attempts=attempts,
                )

            except Exception as e:
                logger.warning(f"RAG generation attempt {attempts} failed: {str(e)}")
                if attempts < max_generation_attempts:
                    time.sleep(2**attempts + random.uniform(0, 1))
                else:
                    # Fallback to direct generation
                    logger.error(
                        "RAG generation failed, falling back to direct generation"
                    )
                    content = self._generate_content_direct(templates)
                    return RAGGenerationResult(
                        content=content,
                        similarity_results={},
                        was_generated=True,
                        generation_attempts=attempts,
                    )

    def _generate_content_direct(
        self, templates: ProcessingTemplates
    ) -> BehavioralContent:
        """Generate content directly without RAG (original implementation)."""
        # Prepare examples for prompt (configurable limits)
        max_examples = int(os.getenv("MAX_PROMPT_EXAMPLES", "3"))
        about_examples = "\n".join(templates.about_templates[:max_examples])
        goals_examples = "\n".join(templates.goals_templates[: max_examples * 2])
        frustrations_examples = "\n".join(
            templates.frustrations_templates[: max_examples * 2]
        )
        need_state_examples = "\n".join(templates.need_state_templates[:max_examples])
        occasions_examples = "\n".join(templates.occasions_templates[:max_examples])

        # Generate prompt
        prompt = self.prompt_template.format(
            about_examples=about_examples,
            goals_examples=goals_examples,
            frustrations_examples=frustrations_examples,
            need_state_examples=need_state_examples,
            occasions_examples=occasions_examples,
        )

        # Generate with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                content_text = response.content

                # Parse JSON response
                content_data = self._parse_llm_response(content_text)

                # Basic validation - just check required fields exist
                if self._validate_basic_structure(content_data):
                    return BehavioralContent(**content_data)
                else:
                    logger.warning(
                        f"Content structure validation failed on attempt {attempt + 1}"
                    )

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt + random.uniform(0, 1))

        raise Exception(
            f"Failed to generate valid content after {self.max_retries} attempts"
        )

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON."""
        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("No valid JSON found in response")

    def _validate_basic_structure(self, content_data: Dict[str, Any]) -> bool:
        """Basic validation to ensure required fields exist."""
        required_fields = [
            "about",
            "goalsAndMotivations",
            "frustrations",
            "needState",
            "occasions",
        ]

        # Check all required fields exist
        if not all(field in content_data for field in required_fields):
            logger.warning(f"Missing required fields. Expected: {required_fields}")
            return False

        # Check list fields are actually lists
        list_fields = ["goalsAndMotivations", "frustrations"]
        for field in list_fields:
            if (
                not isinstance(content_data[field], list)
                or len(content_data[field]) == 0
            ):
                logger.warning(f"Field {field} must be a non-empty list")
                return False

        # Check string fields are not empty
        string_fields = ["about", "needState", "occasions"]
        for field in string_fields:
            if (
                not isinstance(content_data[field], str)
                or not content_data[field].strip()
            ):
                logger.warning(f"Field {field} must be a non-empty string")
                return False

        return True

    def get_rag_stats(self) -> Optional[Dict[str, Any]]:
        """Get RAG duplicate detector statistics."""
        if self.rag_detector:
            return self.rag_detector.get_stats()
        return None

    def clear_rag_memory(self) -> bool:
        """Clear all RAG memory."""
        if self.rag_detector:
            return self.rag_detector.clear_all()
        return False

    def set_similarity_threshold(self, threshold: float) -> bool:
        """Update similarity threshold for duplicate detection."""
        if self.rag_detector:
            self.rag_detector.similarity_threshold = threshold
            logger.info(f"Updated similarity threshold to {threshold}")
            return True
        return False


# Backward compatibility - maintain original interface
class LLMContentGenerator(RAGEnhancedLLMGenerator):
    """
    Backward compatible LLM Content Generator.

    This maintains the original interface while providing RAG capabilities.
    """

    def __init__(self, provider: str = None, enable_rag: bool = True):
        """Initialize with RAG enabled by default."""
        super().__init__(provider=provider, enable_rag=enable_rag)

    def generate_content(self, templates: ProcessingTemplates) -> BehavioralContent:
        """Original interface - generate content with optional RAG."""
        result = self.generate_content_with_rag(templates)

        # Log RAG statistics if enabled
        if self.enable_rag and result.similarity_results:
            similar_sections = [
                section
                for section, result_obj in result.similarity_results.items()
                if result_obj.is_similar
            ]
            if similar_sections:
                logger.info(f"Reused similar content for sections: {similar_sections}")

        return result.content


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Test the RAG-enhanced generator
    print("=== Testing RAG-Enhanced LLM Generator ===")

    # Create test templates
    test_templates = ProcessingTemplates(
        about_templates=[
            "Passionate about digital marketing and social media strategy",
            "Enthusiastic about creative content and brand storytelling",
            "Focused on data-driven marketing approaches",
        ],
        goals_templates=[
            "Build a strong personal brand online",
            "Learn advanced analytics tools",
            "Grow professional network",
            "Master video content creation",
            "Develop expertise in SEO",
        ],
        frustrations_templates=[
            "Difficulty measuring ROI on social campaigns",
            "Keeping up with platform algorithm changes",
            "Limited budget for paid advertising",
            "Time constraints for content creation",
        ],
        need_state_templates=[
            "Seeking tools and strategies to optimize social media performance",
            "Looking for cost-effective marketing automation solutions",
            "Needing better analytics and reporting capabilities",
        ],
        occasions_templates=[
            "Active during evening hours and weekends for content planning",
            "Most active during morning hours and lunch breaks",
            "Engages primarily on weekdays during business hours",
        ],
    )

    # Initialize generator with RAG enabled
    generator = RAGEnhancedLLMGenerator(enable_rag=True, similarity_threshold=0.8)

    # Generate multiple profiles to test duplicate detection
    print("\n1. Generating first profile...")
    result1 = generator.generate_content_with_rag(test_templates, profile_id=1)
    print(
        f"   Generated: {result1.was_generated}, Attempts: {result1.generation_attempts}"
    )

    print("\n2. Generating second profile...")
    result2 = generator.generate_content_with_rag(test_templates, profile_id=2)
    print(
        f"   Generated: {result2.was_generated}, Attempts: {result2.generation_attempts}"
    )

    # Show similarity results
    if result2.similarity_results:
        print("\n   Similarity Results:")
        for section, sim_result in result2.similarity_results.items():
            print(
                f"   - {section}: Similar={sim_result.is_similar}, Score={sim_result.similarity_score:.3f}"
            )

    # Show RAG statistics
    print("\n3. RAG Statistics:")
    stats = generator.get_rag_stats()
    if stats:
        print(json.dumps(stats, indent=2))

    print("\n=== RAG Testing Complete ===")
