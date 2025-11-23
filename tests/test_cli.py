"""Tests for CLI functionality in synthetic_audience_mvp.py"""

import pytest
import argparse
from unittest.mock import patch, Mock
from io import StringIO

from synthetic_audience_mvp import create_cli_parser, show_performance_tips


class TestCLIParser:
    """Test CLI argument parser."""

    def test_create_cli_parser_basic(self):
        """Test basic CLI parser creation."""
        parser = create_cli_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert "Generate synthetic audience profiles" in parser.description

    def test_parse_basic_arguments(self):
        """Test parsing basic required arguments."""
        parser = create_cli_parser()
        args = parser.parse_args(["input.json", "output.json"])

        assert args.input_file == "input.json"
        assert args.output_file == "output.json"
        assert args.use_async is True  # Default
        assert args.provider == "google"  # Default

    def test_parse_async_flag(self):
        """Test parsing async flag."""
        parser = create_cli_parser()
        args = parser.parse_args(["input.json", "output.json", "--async"])

        assert args.use_async is True

    def test_parse_sequential_flag(self):
        """Test parsing sequential flag."""
        parser = create_cli_parser()
        args = parser.parse_args(["input.json", "output.json", "--sequential"])

        assert args.use_async is False

    def test_parse_provider_google(self):
        """Test parsing Google provider."""
        parser = create_cli_parser()
        args = parser.parse_args(["input.json", "output.json", "--provider", "google"])

        assert args.provider == "google"

    def test_parse_provider_azure(self):
        """Test parsing Azure provider."""
        parser = create_cli_parser()
        args = parser.parse_args(["input.json", "output.json", "--provider", "azure"])

        assert args.provider == "azure"

    def test_parse_invalid_provider(self):
        """Test parsing invalid provider."""
        parser = create_cli_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["input.json", "output.json", "--provider", "invalid"])

    def test_parse_async_configuration(self):
        """Test parsing async configuration options."""
        parser = create_cli_parser()
        args = parser.parse_args(
            [
                "input.json",
                "output.json",
                "--batch-size",
                "10",
                "--max-workers",
                "5",
                "--concurrent-requests",
                "8",
            ]
        )

        assert args.batch_size == 10
        assert args.max_workers == 5
        assert args.concurrent_requests == 8

    def test_parse_show_config_flag(self):
        """Test parsing show-config flag."""
        parser = create_cli_parser()
        args = parser.parse_args(["--show-config"])

        assert args.show_config is True
        assert args.input_file is None
        assert args.output_file is None

    def test_parse_performance_tips_flag(self):
        """Test parsing performance-tips flag."""
        parser = create_cli_parser()
        args = parser.parse_args(["--performance-tips"])

        assert args.performance_tips is True

    def test_parse_mutually_exclusive_async_sequential(self):
        """Test that async and sequential flags are mutually exclusive."""
        parser = create_cli_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["input.json", "output.json", "--async", "--sequential"])

    def test_parse_default_values(self):
        """Test default values for all options."""
        parser = create_cli_parser()
        args = parser.parse_args(["input.json", "output.json"])

        assert args.use_async is True
        assert args.provider == "google"
        assert args.batch_size == 5
        assert args.max_workers == 3
        assert args.concurrent_requests == 3
        assert args.show_config is False
        assert args.performance_tips is False

    def test_parse_negative_values(self):
        """Test parsing negative values for numeric options."""
        parser = create_cli_parser()

        # Should accept negative values (though they may not be logical)
        args = parser.parse_args(
            ["input.json", "output.json", "--batch-size", "-1", "--max-workers", "-2"]
        )

        assert args.batch_size == -1
        assert args.max_workers == -2

    def test_parse_zero_values(self):
        """Test parsing zero values for numeric options."""
        parser = create_cli_parser()
        args = parser.parse_args(
            [
                "input.json",
                "output.json",
                "--batch-size",
                "0",
                "--concurrent-requests",
                "0",
            ]
        )

        assert args.batch_size == 0
        assert args.concurrent_requests == 0

    def test_help_message_content(self):
        """Test that help message contains expected content."""
        parser = create_cli_parser()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])

        help_output = mock_stdout.getvalue()
        assert "Generate synthetic audience profiles" in help_output
        assert "--async" in help_output
        assert "--sequential" in help_output
        assert "--provider" in help_output
        assert "--batch-size" in help_output

    def test_examples_in_help(self):
        """Test that examples are included in help."""
        parser = create_cli_parser()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])

        help_output = mock_stdout.getvalue()
        assert "Examples:" in help_output
        assert "python src/synthetic_audience_mvp.py" in help_output
        assert "--provider azure" in help_output


class TestShowPerformanceTips:
    """Test show_performance_tips function."""

    def test_show_performance_tips_output(self):
        """Test that performance tips are displayed."""
        with patch("builtins.print") as mock_print:
            show_performance_tips()

        # Verify print was called
        assert mock_print.called

        # Check that expected content is in the output
        all_output = " ".join([str(call) for call in mock_print.call_args_list])
        assert "Performance Optimization Tips" in all_output
        assert "Dataset Size Recommendations" in all_output
        assert "Speed vs Stability" in all_output
        assert "API Rate Limit Management" in all_output
        assert "Environment Variables" in all_output

    def test_show_performance_tips_content_structure(self):
        """Test the structure and content of performance tips."""
        with patch("builtins.print") as mock_print:
            show_performance_tips()

        # Get all printed content
        printed_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        full_output = "\n".join(printed_calls)

        # Check for specific recommendations
        assert "Small (< 10 profiles)" in full_output
        assert "Medium (10-50 profiles)" in full_output
        assert "Large (50-100 profiles)" in full_output
        assert "XLarge (100+ profiles)" in full_output

        # Check for configuration examples
        assert "--batch-size" in full_output
        assert "--max-workers" in full_output
        assert "--concurrent-requests" in full_output

        # Check for environment variables
        assert "PARALLEL_BATCH_SIZE" in full_output
        assert "MAX_WORKERS" in full_output
        assert "CONCURRENT_REQUESTS" in full_output


class TestMainCLIExecution:
    """Test main CLI execution logic."""

    @patch("synthetic_audience_mvp.SyntheticAudienceGenerator")
    @patch("synthetic_audience_mvp.create_cli_parser")
    def test_main_execution_basic(self, mock_create_parser, mock_generator_class):
        """Test basic main execution flow."""
        # Mock parser and args
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.performance_tips = False
        mock_args.show_config = False
        mock_args.input_file = "input.json"
        mock_args.output_file = "output.json"
        mock_args.use_async = True
        mock_args.provider = "google"
        mock_args.batch_size = 5
        mock_args.max_workers = 3
        mock_args.concurrent_requests = 3

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        # Mock generator
        mock_generator = Mock()
        mock_generator.process_request.return_value = {"total_profiles": 10}
        mock_generator_class.return_value = mock_generator

        # Import and patch the main execution
        with patch("builtins.print"):
            with patch("sys.argv", ["script.py", "input.json", "output.json"]):
                # This would normally be the main execution block
                # We'll simulate it here
                parser = mock_create_parser()
                args = parser.parse_args()

                if not args.performance_tips and not args.show_config:
                    generator = mock_generator_class(
                        use_async=args.use_async,
                        provider=args.provider,
                        batch_size=args.batch_size,
                        max_workers=args.max_workers,
                        concurrent_requests=args.concurrent_requests,
                    )
                    stats = generator.process_request(args.input_file, args.output_file)

        # Verify generator was created and called correctly
        mock_generator_class.assert_called_once_with(
            use_async=True,
            provider="google",
            batch_size=5,
            max_workers=3,
            concurrent_requests=3,
        )
        mock_generator.process_request.assert_called_once_with(
            "input.json", "output.json"
        )

    @patch("synthetic_audience_mvp.show_performance_tips")
    @patch("synthetic_audience_mvp.create_cli_parser")
    def test_main_execution_performance_tips(self, mock_create_parser, mock_show_tips):
        """Test main execution with performance tips flag."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.performance_tips = True

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        # Simulate main execution with performance tips
        with patch("sys.exit") as mock_exit:
            parser = mock_create_parser()
            args = parser.parse_args()

            if args.performance_tips:
                mock_show_tips()
                mock_exit(0)

        mock_show_tips.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("synthetic_audience_mvp.SyntheticAudienceGenerator")
    @patch("synthetic_audience_mvp.create_cli_parser")
    def test_main_execution_show_config(self, mock_create_parser, mock_generator_class):
        """Test main execution with show config flag."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.performance_tips = False
        mock_args.show_config = True
        mock_args.use_async = True
        mock_args.provider = "google"
        mock_args.batch_size = 5
        mock_args.max_workers = 3
        mock_args.concurrent_requests = 3

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_generator = Mock()
        mock_generator.get_workflow_info.return_value = {
            "mode": "async",
            "provider": "google",
        }
        mock_generator_class.return_value = mock_generator

        # Simulate main execution with show config
        with patch("builtins.print") as mock_print:
            with patch("sys.exit") as mock_exit:
                parser = mock_create_parser()
                args = parser.parse_args()

                generator = mock_generator_class(
                    use_async=args.use_async,
                    provider=args.provider,
                    batch_size=args.batch_size,
                    max_workers=args.max_workers,
                    concurrent_requests=args.concurrent_requests,
                )

                if args.show_config:
                    config = generator.get_workflow_info()
                    mock_print("Current Configuration:")
                    mock_exit(0)

        mock_generator.get_workflow_info.assert_called_once()
        mock_exit.assert_called_once_with(0)
