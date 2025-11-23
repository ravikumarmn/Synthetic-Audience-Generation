"""Tests for SyntheticAudienceGenerator main class in synthetic_audience_mvp.py"""

import pytest
import os
import time
from unittest.mock import patch, Mock, MagicMock

from synthetic_audience_mvp import (
    SyntheticAudienceGenerator,
    create_async_workflow,
    create_sequential_workflow,
    create_parallel_workflow,
)


class TestSyntheticAudienceGenerator:
    """Test SyntheticAudienceGenerator class."""

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        generator = SyntheticAudienceGenerator()

        assert generator.use_async is True
        assert generator.provider == "google"  # From test environment
        assert generator.batch_size == 3  # From test environment
        assert generator.max_workers == 2  # From test environment
        assert generator.concurrent_requests == 2  # From test environment

    def test_initialization_custom_values(self):
        """Test initialization with custom values."""
        generator = SyntheticAudienceGenerator(
            use_async=False,
            provider="azure",
            batch_size=10,
            max_workers=5,
            concurrent_requests=8,
        )

        assert generator.use_async is False
        assert generator.provider == "azure"
        assert generator.batch_size == 10
        assert generator.max_workers == 5
        assert generator.concurrent_requests == 8

    @patch("synthetic_audience_mvp.create_async_workflow")
    def test_initialization_async_workflow(self, mock_create_async):
        """Test that async workflow is created for async mode."""
        mock_workflow = Mock()
        mock_workflow.compile.return_value = Mock()
        mock_create_async.return_value = mock_workflow

        generator = SyntheticAudienceGenerator(use_async=True)

        mock_create_async.assert_called_once()
        assert generator.workflow == mock_workflow

    @patch("synthetic_audience_mvp.create_sequential_workflow")
    def test_initialization_sequential_workflow(self, mock_create_sequential):
        """Test that sequential workflow is created for sequential mode."""
        mock_workflow = Mock()
        mock_workflow.compile.return_value = Mock()
        mock_create_sequential.return_value = mock_workflow

        generator = SyntheticAudienceGenerator(use_async=False)

        mock_create_sequential.assert_called_once()
        assert generator.workflow == mock_workflow

    def test_get_workflow_info_async(self):
        """Test workflow info for async mode."""
        generator = SyntheticAudienceGenerator(
            use_async=True,
            provider="azure",
            batch_size=10,
            max_workers=5,
            concurrent_requests=8,
        )

        info = generator.get_workflow_info()

        assert info["mode"] == "async"
        assert info["provider"] == "azure"
        assert info["batch_size"] == 10
        assert info["max_workers"] == 5
        assert info["concurrent_requests"] == 8

    def test_get_workflow_info_sequential(self):
        """Test workflow info for sequential mode."""
        generator = SyntheticAudienceGenerator(use_async=False, provider="google")

        info = generator.get_workflow_info()

        assert info["mode"] == "sequential"
        assert info["provider"] == "google"
        assert info["batch_size"] == "N/A"
        assert info["max_workers"] == "N/A"
        assert info["concurrent_requests"] == "N/A"

    @patch("synthetic_audience_mvp.create_async_workflow")
    def test_process_request_success(
        self, mock_create_async, temp_input_file, temp_output_file
    ):
        """Test successful request processing."""
        # Mock the workflow and app
        mock_app = Mock()
        mock_final_state = {
            "generation_stats": {
                "total_profiles": 10,
                "generation_timestamp": "2023-01-01 12:00:00",
                "processing_mode": "async",
            }
        }
        mock_app.invoke.return_value = mock_final_state

        mock_workflow = Mock()
        mock_workflow.compile.return_value = mock_app
        mock_create_async.return_value = mock_workflow

        generator = SyntheticAudienceGenerator(use_async=True)

        with patch("builtins.print"):  # Suppress print output
            with patch("time.time", side_effect=[0, 2.5]):  # Mock timing
                stats = generator.process_request(temp_input_file, temp_output_file)

        # Verify app was called with correct initial state
        mock_app.invoke.assert_called_once()
        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["input_file"] == temp_input_file
        assert call_args["output_file"] == temp_output_file
        assert call_args["use_async"] is True
        assert call_args["provider"] == "google"

        # Verify stats include processing time
        assert "total_processing_time" in stats
        assert stats["total_processing_time"] == "2.50 seconds"
        assert stats["total_profiles"] == 10

    @patch("synthetic_audience_mvp.create_sequential_workflow")
    def test_process_request_sequential_mode(
        self, mock_create_sequential, temp_input_file, temp_output_file
    ):
        """Test request processing in sequential mode."""
        mock_app = Mock()
        mock_final_state = {
            "generation_stats": {"total_profiles": 5, "processing_mode": "sequential"}
        }
        mock_app.invoke.return_value = mock_final_state

        mock_workflow = Mock()
        mock_workflow.compile.return_value = mock_app
        mock_create_sequential.return_value = mock_workflow

        generator = SyntheticAudienceGenerator(use_async=False, provider="azure")

        with patch("builtins.print"):
            with patch("time.time", side_effect=[0, 1.0]):
                stats = generator.process_request(temp_input_file, temp_output_file)

        # Verify correct parameters were passed
        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["use_async"] is False
        assert call_args["provider"] == "azure"

    @patch("synthetic_audience_mvp.create_async_workflow")
    def test_process_request_with_exception(
        self, mock_create_async, temp_input_file, temp_output_file
    ):
        """Test request processing with exception."""
        mock_app = Mock()
        mock_app.invoke.side_effect = Exception("Processing error")

        mock_workflow = Mock()
        mock_workflow.compile.return_value = mock_app
        mock_create_async.return_value = mock_workflow

        generator = SyntheticAudienceGenerator()

        with pytest.raises(Exception, match="Processing error"):
            generator.process_request(temp_input_file, temp_output_file)

    @patch("synthetic_audience_mvp.create_async_workflow")
    def test_process_request_missing_stats(
        self, mock_create_async, temp_input_file, temp_output_file
    ):
        """Test request processing when stats are missing from final state."""
        mock_app = Mock()
        mock_final_state = {}  # No generation_stats
        mock_app.invoke.return_value = mock_final_state

        mock_workflow = Mock()
        mock_workflow.compile.return_value = mock_app
        mock_create_async.return_value = mock_workflow

        generator = SyntheticAudienceGenerator()

        with patch("builtins.print"):
            with patch("time.time", side_effect=[0, 1.0]):
                stats = generator.process_request(temp_input_file, temp_output_file)

        # Should handle missing stats gracefully
        assert "total_processing_time" in stats
        assert stats["total_processing_time"] == "1.00 seconds"

    def test_environment_variable_integration(self):
        """Test that environment variables are properly used."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "azure",
                "PARALLEL_BATCH_SIZE": "15",
                "MAX_WORKERS": "8",
                "CONCURRENT_REQUESTS": "10",
            },
        ):
            generator = SyntheticAudienceGenerator()

            assert generator.provider == "azure"
            assert generator.batch_size == 15
            assert generator.max_workers == 8
            assert generator.concurrent_requests == 10

    def test_parameter_override_environment(self):
        """Test that constructor parameters override environment variables."""
        with patch.dict(
            os.environ, {"LLM_PROVIDER": "azure", "PARALLEL_BATCH_SIZE": "15"}
        ):
            generator = SyntheticAudienceGenerator(provider="google", batch_size=5)

            # Constructor parameters should override environment
            assert generator.provider == "google"
            assert generator.batch_size == 5


class TestWorkflowCreationFunctions:
    """Test workflow creation functions."""

    def test_create_async_workflow(self):
        """Test async workflow creation."""
        workflow = create_async_workflow()

        # Should be a StateGraph
        assert hasattr(workflow, "add_node")
        assert hasattr(workflow, "add_edge")
        assert hasattr(workflow, "compile")

    def test_create_sequential_workflow(self):
        """Test sequential workflow creation."""
        workflow = create_sequential_workflow()

        # Should be a StateGraph
        assert hasattr(workflow, "add_node")
        assert hasattr(workflow, "add_edge")
        assert hasattr(workflow, "compile")

    def test_create_parallel_workflow_legacy(self):
        """Test legacy parallel workflow function."""
        # Should be an alias for create_async_workflow
        async_workflow = create_async_workflow()
        parallel_workflow = create_parallel_workflow()

        # Both should have the same structure (this is a basic check)
        assert type(async_workflow) == type(parallel_workflow)

    @patch("synthetic_audience_mvp.StateGraph")
    def test_async_workflow_nodes(self, mock_state_graph):
        """Test that async workflow has correct nodes."""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow

        create_async_workflow()

        # Verify nodes were added
        expected_nodes = [
            "load_json",
            "distribution_builder",
            "persona_processor",
            "generate_profiles_async",
            "finalize_profiles",
            "output_writer",
        ]

        # Check that add_node was called for each expected node
        add_node_calls = [call[0][0] for call in mock_workflow.add_node.call_args_list]
        for node in expected_nodes:
            assert node in add_node_calls

    @patch("synthetic_audience_mvp.StateGraph")
    def test_sequential_workflow_nodes(self, mock_state_graph):
        """Test that sequential workflow has correct nodes."""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow

        create_sequential_workflow()

        # Verify nodes were added
        expected_nodes = [
            "load_json",
            "distribution_builder",
            "persona_processor",
            "generate_profiles_sequential",
            "finalize_profiles",
            "output_writer",
        ]

        # Check that add_node was called for each expected node
        add_node_calls = [call[0][0] for call in mock_workflow.add_node.call_args_list]
        for node in expected_nodes:
            assert node in add_node_calls
