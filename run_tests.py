#!/usr/bin/env python3
"""
Test runner script for synthetic audience MVP tests.
Provides various test execution options and reporting.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for Synthetic Audience MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --fast             # Run fast tests only
  python run_tests.py --file test_models # Run specific test file
  python run_tests.py --verbose          # Verbose output
        """,
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (exclude integration tests)",
    )
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run fast tests only (exclude slow tests)"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage report"
    )
    parser.add_argument(
        "--file", type=str, help="Run specific test file (e.g., test_models)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Number of parallel test processes",
    )
    parser.add_argument(
        "--lf",
        "--last-failed",
        action="store_true",
        help="Run only tests that failed in the last run",
    )
    parser.add_argument(
        "--pdb", action="store_true", help="Drop into debugger on failures"
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test selection options
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.fast:
        cmd.extend(["-m", "not slow"])

    # Add specific file if requested
    if args.file:
        test_file = f"tests/test_{args.file}.py"
        if not os.path.exists(test_file):
            test_file = f"tests/{args.file}.py"
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {args.file}")
            return False
        cmd.append(test_file)

    # Add coverage if requested
    if args.coverage:
        cmd.extend(
            [
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ]
        )

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")

    # Add parallel execution
    if args.parallel > 1:
        cmd.extend(["-n", str(args.parallel)])

    # Add last failed
    if args.lf:
        cmd.append("--lf")

    # Add debugger
    if args.pdb:
        cmd.append("--pdb")

    # Run the tests
    success = run_command(cmd, "Running Synthetic Audience MVP Tests")

    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        return False

    return True


def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = ["pytest", "pytest-asyncio", "pytest-cov", "pytest-xdist"]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing test dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    return True


if __name__ == "__main__":
    print("🧪 Synthetic Audience MVP Test Runner")
    print("=" * 60)

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Run tests
    success = main()
    sys.exit(0 if success else 1)
