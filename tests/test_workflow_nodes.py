"""Tests for workflow nodes in synthetic_audience_mvp.py"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, Mock

from synthetic_audience_mvp import (
    load_json_node,
    distribution_builder_node,
    persona_processor_node,
    generate_profiles_sequential,
    generate_profiles_async,
    finalize_profiles_node,
    output_writer_node,
    FilterDetails,
    InputPersona,
    RequestData,
    ProcessingTemplates,
    DemographicAssignment,
    SyntheticProfile,
)


class TestLoadJsonNode:
    """Test load_json_node function."""

    def test_load_json_node_success(self, temp_input_file):
        """Test successful JSON loading."""
        state = {"input_file": temp_input_file}
        result = load_json_node(state)

        assert "filter_details" in result
        assert "input_personas" in result
        assert isinstance(result["filter_details"], FilterDetails)
        assert isinstance(result["input_personas"], list)
        assert len(result["input_personas"]) == 2
        assert all(isinstance(p, InputPersona) for p in result["input_personas"])

    def test_load_json_node_file_not_found(self):
        """Test JSON loading with non-existent file."""
        state = {"input_file": "/non/existent/file.json"}

        with pytest.raises(FileNotFoundError):
            load_json_node(state)

    def test_load_json_node_invalid_json(self):
        """Test JSON loading with invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("Invalid JSON content")
            temp_file = f.name

        try:
            state = {"input_file": temp_file}
            with pytest.raises(json.JSONDecodeError):
                load_json_node(state)
        finally:
            os.unlink(temp_file)

    def test_load_json_node_invalid_structure(self):
        """Test JSON loading with invalid data structure."""
        invalid_data = {"invalid": "structure"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = f.name

        try:
            state = {"input_file": temp_file}
            with pytest.raises(Exception):  # Should raise validation error
                load_json_node(state)
        finally:
            os.unlink(temp_file)


class TestDistributionBuilderNode:
    """Test distribution_builder_node function."""

    def test_distribution_builder_node_success(self, sample_filter_details):
        """Test successful distribution building."""
        state = {"filter_details": sample_filter_details}
        result = distribution_builder_node(state)

        assert "demographic_schedule" in result
        assert isinstance(result["demographic_schedule"], list)
        assert (
            len(result["demographic_schedule"])
            == sample_filter_details.user_req_responses
        )
        assert all(
            isinstance(d, DemographicAssignment) for d in result["demographic_schedule"]
        )

    def test_distribution_builder_node_preserves_state(self, sample_filter_details):
        """Test that distribution builder preserves existing state."""
        original_state = {
            "filter_details": sample_filter_details,
            "input_file": "test.json",
            "output_file": "output.json",
        }
        result = distribution_builder_node(original_state)

        # Should preserve original state
        assert result["input_file"] == "test.json"
        assert result["output_file"] == "output.json"
        assert result["filter_details"] == sample_filter_details
        # Should add demographic_schedule
        assert "demographic_schedule" in result


class TestPersonaProcessorNode:
    """Test persona_processor_node function."""

    def test_persona_processor_node_success(self, sample_input_personas):
        """Test successful persona processing."""
        state = {"input_personas": sample_input_personas}
        result = persona_processor_node(state)

        assert "processing_templates" in result
        assert isinstance(result["processing_templates"], ProcessingTemplates)
        assert len(result["processing_templates"].about_templates) > 0
        assert len(result["processing_templates"].goals_templates) > 0

    def test_persona_processor_node_empty_personas(self):
        """Test persona processing with empty personas list."""
        state = {"input_personas": []}
        result = persona_processor_node(state)

        assert "processing_templates" in result
        templates = result["processing_templates"]
        assert len(templates.about_templates) == 0
        assert len(templates.goals_templates) == 0


class TestGenerateProfilesSequential:
    """Test generate_profiles_sequential function."""

    @patch("synthetic_audience_mvp.LLMContentGenerator")
    def test_generate_profiles_sequential_success(
        self,
        mock_generator_class,
        sample_demographic_assignments,
        sample_processing_templates,
        sample_behavioral_content,
    ):
        """Test successful sequential profile generation."""
        mock_generator = Mock()
        mock_generator.generate_content.return_value = sample_behavioral_content
        mock_generator_class.return_value = mock_generator

        state = {
            "demographic_schedule": sample_demographic_assignments,
            "processing_templates": sample_processing_templates,
            "provider": "google",
        }

        with patch("builtins.print"):  # Suppress print output
            result = generate_profiles_sequential(state)

        assert "all_profiles" in result
        assert len(result["all_profiles"]) == len(sample_demographic_assignments)
        assert all(isinstance(p, SyntheticProfile) for p in result["all_profiles"])

        # Verify generator was called correct number of times
        assert mock_generator.generate_content.call_count == len(
            sample_demographic_assignments
        )

    @patch("synthetic_audience_mvp.LLMContentGenerator")
    def test_generate_profiles_sequential_with_exception(
        self,
        mock_generator_class,
        sample_demographic_assignments,
        sample_processing_templates,
    ):
        """Test sequential generation with LLM exception."""
        mock_generator = Mock()
        mock_generator.generate_content.side_effect = Exception("LLM Error")
        mock_generator_class.return_value = mock_generator

        state = {
            "demographic_schedule": sample_demographic_assignments,
            "processing_templates": sample_processing_templates,
        }

        with pytest.raises(Exception, match="LLM Error"):
            generate_profiles_sequential(state)

    @patch("synthetic_audience_mvp.LLMContentGenerator")
    def test_generate_profiles_sequential_empty_schedule(
        self, mock_generator_class, sample_processing_templates
    ):
        """Test sequential generation with empty demographic schedule."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        state = {
            "demographic_schedule": [],
            "processing_templates": sample_processing_templates,
        }

        with patch("builtins.print"):
            result = generate_profiles_sequential(state)

        assert result["all_profiles"] == []
        mock_generator.generate_content.assert_not_called()


class TestGenerateProfilesAsync:
    """Test generate_profiles_async function."""

    @patch("synthetic_audience_mvp.ParallelBatchProcessor")
    @patch("asyncio.run")
    def test_generate_profiles_async_success(
        self,
        mock_asyncio_run,
        mock_processor_class,
        sample_demographic_assignments,
        sample_processing_templates,
        sample_synthetic_profile,
    ):
        """Test successful async profile generation."""
        # Mock the processor
        mock_processor = Mock()
        mock_processor.process_profiles_parallel.return_value = [
            sample_synthetic_profile
        ] * 3
        mock_processor_class.return_value = mock_processor

        # Mock asyncio.run to return the mocked result
        mock_asyncio_run.return_value = [sample_synthetic_profile] * 3

        state = {
            "demographic_schedule": sample_demographic_assignments,
            "processing_templates": sample_processing_templates,
            "provider": "google",
            "batch_size": 5,
            "max_workers": 3,
            "concurrent_requests": 3,
        }

        with patch("builtins.print"):  # Suppress print output
            with patch("time.time", side_effect=[0, 1.5]):  # Mock timing
                result = generate_profiles_async(state)

        assert "all_profiles" in result
        assert len(result["all_profiles"]) == 3

        # Verify processor was created with correct parameters
        mock_processor_class.assert_called_once_with(
            provider="google", batch_size=5, max_workers=3, concurrent_requests=3
        )

    @patch("synthetic_audience_mvp.ParallelBatchProcessor")
    @patch("asyncio.run")
    def test_generate_profiles_async_default_config(
        self,
        mock_asyncio_run,
        mock_processor_class,
        sample_demographic_assignments,
        sample_processing_templates,
    ):
        """Test async generation with default configuration."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_asyncio_run.return_value = []

        state = {
            "demographic_schedule": sample_demographic_assignments,
            "processing_templates": sample_processing_templates,
        }

        with patch("builtins.print"):
            with patch("time.time", side_effect=[0, 1]):
                result = generate_profiles_async(state)

        # Should use environment defaults (from conftest.py)
        mock_processor_class.assert_called_once_with(
            provider="google",
            batch_size=3,  # From test environment
            max_workers=2,  # From test environment
            concurrent_requests=2,  # From test environment
        )


class TestFinalizeProfilesNode:
    """Test finalize_profiles_node function."""

    def test_finalize_profiles_node_success(self, sample_synthetic_profile):
        """Test successful profile finalization."""
        profiles = [sample_synthetic_profile]
        profiles[0].profile_id = 2  # Test sorting

        state = {
            "all_profiles": profiles,
            "use_async": True,
            "batch_size": 5,
            "max_workers": 3,
            "concurrent_requests": 3,
        }

        with patch("time.strftime", return_value="2023-01-01 12:00:00"):
            result = finalize_profiles_node(state)

        assert "generation_stats" in result
        stats = result["generation_stats"]
        assert stats["total_profiles"] == 1
        assert stats["generation_timestamp"] == "2023-01-01 12:00:00"
        assert stats["processing_mode"] == "async"
        assert stats["batch_size"] == 5
        assert stats["max_workers"] == 3
        assert stats["concurrent_requests"] == 3

    def test_finalize_profiles_node_sorting(self):
        """Test that profiles are sorted by ID."""
        profiles = [
            SyntheticProfile(
                age_bucket="18-25",
                gender="Male",
                ethnicity="White",
                about="Test",
                goalsAndMotivations=[],
                frustrations=[],
                needState="Test",
                occasions="Test",
                profile_id=3,
            ),
            SyntheticProfile(
                age_bucket="26-35",
                gender="Female",
                ethnicity="Hispanic",
                about="Test",
                goalsAndMotivations=[],
                frustrations=[],
                needState="Test",
                occasions="Test",
                profile_id=1,
            ),
            SyntheticProfile(
                age_bucket="36-45",
                gender="Male",
                ethnicity="Black",
                about="Test",
                goalsAndMotivations=[],
                frustrations=[],
                needState="Test",
                occasions="Test",
                profile_id=2,
            ),
        ]

        state = {"all_profiles": profiles, "use_async": False}

        with patch("time.strftime", return_value="2023-01-01 12:00:00"):
            result = finalize_profiles_node(state)

        sorted_profiles = result["all_profiles"]
        profile_ids = [p.profile_id for p in sorted_profiles]
        assert profile_ids == [1, 2, 3]

    def test_finalize_profiles_node_sequential_mode(self):
        """Test finalization with sequential mode."""
        state = {"all_profiles": [], "use_async": False}

        with patch("time.strftime", return_value="2023-01-01 12:00:00"):
            result = finalize_profiles_node(state)

        stats = result["generation_stats"]
        assert stats["processing_mode"] == "sequential"
        assert stats["batch_size"] == "N/A"
        assert stats["max_workers"] == "N/A"
        assert stats["concurrent_requests"] == "N/A"


class TestOutputWriterNode:
    """Test output_writer_node function."""

    def test_output_writer_node_success(
        self, temp_output_file, sample_synthetic_profile
    ):
        """Test successful output writing."""
        generation_stats = {
            "total_profiles": 1,
            "generation_timestamp": "2023-01-01 12:00:00",
            "processing_mode": "async",
        }

        state = {
            "all_profiles": [sample_synthetic_profile],
            "generation_stats": generation_stats,
            "output_file": temp_output_file,
        }

        result = output_writer_node(state)

        assert result["output_written"] is True

        # Verify file was written correctly
        with open(temp_output_file, "r") as f:
            output_data = json.load(f)

        assert "synthetic_audience" in output_data
        assert "generation_metadata" in output_data
        assert len(output_data["synthetic_audience"]) == 1
        assert output_data["generation_metadata"]["total_profiles"] == 1

    def test_output_writer_node_default_filename(self, sample_synthetic_profile):
        """Test output writing with default filename."""
        state = {
            "all_profiles": [sample_synthetic_profile],
            "generation_stats": {"total_profiles": 1},
        }

        # Mock the file writing to avoid creating actual file
        with patch("builtins.open", create=True) as mock_open:
            with patch("json.dump") as mock_json_dump:
                result = output_writer_node(state)

        assert result["output_written"] is True
        # Should use default filename
        mock_open.assert_called_once_with("synthetic_audience.json", "w")

    def test_output_writer_node_empty_profiles(self, temp_output_file):
        """Test output writing with empty profiles list."""
        state = {
            "all_profiles": [],
            "generation_stats": {"total_profiles": 0},
            "output_file": temp_output_file,
        }

        result = output_writer_node(state)

        assert result["output_written"] is True

        # Verify empty output
        with open(temp_output_file, "r") as f:
            output_data = json.load(f)

        assert output_data["synthetic_audience"] == []
        assert output_data["generation_metadata"]["total_profiles"] == 0
