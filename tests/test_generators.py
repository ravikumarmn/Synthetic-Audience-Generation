"""Tests for LLM content generators in synthetic_audience_mvp.py"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from synthetic_audience_mvp import (
    LLMContentGenerator,
    AsyncLLMContentGenerator,
    ParallelBatchProcessor,
    BehavioralContent,
    ProcessingTemplates,
    DemographicAssignment,
    SyntheticProfile,
)


class TestLLMContentGenerator:
    """Test LLMContentGenerator class."""

    @patch("synthetic_audience_mvp.ChatGoogleGenerativeAI")
    def test_google_provider_initialization(self, mock_google_ai):
        """Test initialization with Google provider."""
        generator = LLMContentGenerator(provider="google")
        assert generator.provider == "google"
        mock_google_ai.assert_called_once()

    @patch("synthetic_audience_mvp.AzureChatOpenAI")
    def test_azure_provider_initialization(self, mock_azure_ai):
        """Test initialization with Azure provider."""
        generator = LLMContentGenerator(provider="azure")
        assert generator.provider == "azure"
        mock_azure_ai.assert_called_once()

    @patch("synthetic_audience_mvp.ChatGoogleGenerativeAI")
    def test_default_provider_initialization(self, mock_google_ai):
        """Test initialization with default provider."""
        generator = LLMContentGenerator()
        assert generator.provider == "google"
        mock_google_ai.assert_called_once()

    @patch("synthetic_audience_mvp.ChatGoogleGenerativeAI")
    def test_generate_content_success(
        self, mock_google_ai, sample_processing_templates, mock_llm_response
    ):
        """Test successful content generation."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_google_ai.return_value = mock_llm

        generator = LLMContentGenerator(provider="google")
        result = generator.generate_content(sample_processing_templates)

        assert isinstance(result, BehavioralContent)
        assert "technology" in result.about
        assert len(result.goalsAndMotivations) == 3
        assert len(result.frustrations) == 3
        mock_llm.invoke.assert_called_once()

    @patch("synthetic_audience_mvp.ChatGoogleGenerativeAI")
    def test_generate_content_invalid_json(
        self, mock_google_ai, sample_processing_templates
    ):
        """Test content generation with invalid JSON response."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Invalid JSON content"
        mock_llm.invoke.return_value = mock_response
        mock_google_ai.return_value = mock_llm

        generator = LLMContentGenerator(provider="google")

        with pytest.raises(json.JSONDecodeError):
            generator.generate_content(sample_processing_templates)

    @patch("synthetic_audience_mvp.ChatGoogleGenerativeAI")
    def test_prompt_template_formatting(
        self, mock_google_ai, sample_processing_templates
    ):
        """Test that prompt template is properly formatted."""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = json.dumps(
            {
                "about": "Test about",
                "goalsAndMotivations": ["Goal 1"],
                "frustrations": ["Frustration 1"],
                "needState": "Test need state",
                "occasions": "Test occasions",
            }
        )
        mock_google_ai.return_value = mock_llm

        generator = LLMContentGenerator(provider="google")
        generator.generate_content(sample_processing_templates)

        # Verify that invoke was called with a formatted prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Tech enthusiast" in call_args  # From about_templates
        assert "Learn new technologies" in call_args  # From goals_templates


class TestAsyncLLMContentGenerator:
    """Test AsyncLLMContentGenerator class."""

    @patch("synthetic_audience_mvp.LLMContentGenerator")
    def test_initialization(self, mock_generator_class):
        """Test async generator initialization."""
        async_gen = AsyncLLMContentGenerator(provider="google", max_workers=5)
        assert async_gen.max_workers == 5
        mock_generator_class.assert_called_once_with("google")

    @patch("synthetic_audience_mvp.LLMContentGenerator")
    @pytest.mark.asyncio
    async def test_generate_content_async(
        self,
        mock_generator_class,
        sample_processing_templates,
        sample_behavioral_content,
    ):
        """Test async content generation."""
        mock_generator = Mock()
        mock_generator.generate_content.return_value = sample_behavioral_content
        mock_generator_class.return_value = mock_generator

        async_gen = AsyncLLMContentGenerator(provider="google", max_workers=2)
        result = await async_gen.generate_content_async(sample_processing_templates)

        assert isinstance(result, BehavioralContent)
        assert result.about == sample_behavioral_content.about
        mock_generator.generate_content.assert_called_once_with(
            sample_processing_templates
        )

    @patch("synthetic_audience_mvp.LLMContentGenerator")
    @pytest.mark.asyncio
    async def test_generate_content_async_exception(
        self, mock_generator_class, sample_processing_templates
    ):
        """Test async content generation with exception."""
        mock_generator = Mock()
        mock_generator.generate_content.side_effect = Exception("API Error")
        mock_generator_class.return_value = mock_generator

        async_gen = AsyncLLMContentGenerator(provider="google")

        with pytest.raises(Exception, match="API Error"):
            await async_gen.generate_content_async(sample_processing_templates)


class TestParallelBatchProcessor:
    """Test ParallelBatchProcessor class."""

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        processor = ParallelBatchProcessor()
        assert processor.provider is None
        assert processor.batch_size == 3  # From test environment
        assert processor.max_workers == 2  # From test environment
        assert processor.concurrent_requests == 2  # From test environment

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        processor = ParallelBatchProcessor(
            provider="azure", batch_size=10, max_workers=5, concurrent_requests=8
        )
        assert processor.provider == "azure"
        assert processor.batch_size == 10
        assert processor.max_workers == 5
        assert processor.concurrent_requests == 8

    @patch("synthetic_audience_mvp.AsyncLLMContentGenerator")
    @pytest.mark.asyncio
    async def test_process_profiles_parallel_success(
        self,
        mock_async_gen_class,
        sample_demographic_assignments,
        sample_processing_templates,
        sample_behavioral_content,
    ):
        """Test successful parallel profile processing."""
        # Mock the async generator
        mock_async_gen = Mock()

        # Create a proper async mock that returns the behavioral content
        async def mock_generate(*args, **kwargs):
            return sample_behavioral_content

        mock_async_gen.generate_content_async = mock_generate
        mock_async_gen_class.return_value = mock_async_gen

        processor = ParallelBatchProcessor(batch_size=2, concurrent_requests=2)
        results = await processor.process_profiles_parallel(
            sample_demographic_assignments, sample_processing_templates
        )

        assert len(results) == 3  # Same as input demographics
        assert all(isinstance(profile, SyntheticProfile) for profile in results)
        assert results[0].profile_id == 1  # profile_index + 1
        assert results[1].profile_id == 2
        assert results[2].profile_id == 3

    @patch("synthetic_audience_mvp.AsyncLLMContentGenerator")
    @pytest.mark.asyncio
    async def test_process_profiles_parallel_with_exceptions(
        self,
        mock_async_gen_class,
        sample_demographic_assignments,
        sample_processing_templates,
        sample_behavioral_content,
    ):
        """Test parallel processing with some exceptions."""
        # Mock the async generator to fail on second call
        mock_async_gen = Mock()

        async def mock_generate_success():
            return sample_behavioral_content

        async def mock_generate_failure():
            raise Exception("API Error")

        # Set up side effects for different calls
        call_count = 0

        async def mock_generate_side_effect(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise Exception("API Error")
            return sample_behavioral_content

        mock_async_gen.generate_content_async = mock_generate_side_effect
        mock_async_gen_class.return_value = mock_async_gen

        processor = ParallelBatchProcessor(batch_size=2, concurrent_requests=2)

        # Capture print output to verify error handling
        with patch("builtins.print") as mock_print:
            results = await processor.process_profiles_parallel(
                sample_demographic_assignments, sample_processing_templates
            )

        # Should have 2 successful results (1 failed)
        assert len(results) == 2
        assert all(isinstance(profile, SyntheticProfile) for profile in results)

        # Verify error was printed
        mock_print.assert_called()
        error_calls = [
            call
            for call in mock_print.call_args_list
            if "Error generating profile" in str(call)
        ]
        assert len(error_calls) == 1

    @patch("synthetic_audience_mvp.AsyncLLMContentGenerator")
    @pytest.mark.asyncio
    async def test_process_profiles_parallel_empty_input(
        self, mock_async_gen_class, sample_processing_templates
    ):
        """Test parallel processing with empty demographics list."""
        processor = ParallelBatchProcessor()
        results = await processor.process_profiles_parallel(
            [], sample_processing_templates
        )

        assert results == []
        # Async generator should not be created for empty input
        mock_async_gen_class.assert_called_once()

    @patch("synthetic_audience_mvp.AsyncLLMContentGenerator")
    @pytest.mark.asyncio
    async def test_batch_processing_delay(
        self,
        mock_async_gen_class,
        sample_processing_templates,
        sample_behavioral_content,
    ):
        """Test that delays are added between batches."""
        # Create more demographics to test batching
        demographics = [
            DemographicAssignment(
                age_bucket="18-25", gender="Male", ethnicity="White", profile_index=i
            )
            for i in range(5)
        ]

        mock_async_gen = Mock()

        async def mock_generate(*args, **kwargs):
            return sample_behavioral_content

        mock_async_gen.generate_content_async = mock_generate
        mock_async_gen_class.return_value = mock_async_gen

        processor = ParallelBatchProcessor(batch_size=2, concurrent_requests=2)

        with patch("asyncio.sleep") as mock_sleep:
            results = await processor.process_profiles_parallel(
                demographics, sample_processing_templates
            )

        assert len(results) == 5
        # Should have called sleep between batches (5 items, batch_size=2 means 3 batches, 2 sleeps)
        assert mock_sleep.call_count == 2
