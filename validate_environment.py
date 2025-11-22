#!/usr/bin/env python3
"""
Environment validation script for Synthetic Audience Generator MVP
Validates all dependencies and API connectivity before implementation
"""

import sys
import os
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    return (
        False,
        f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+",
    )


def check_package_imports() -> Dict[str, Tuple[bool, str]]:
    """Check if all required packages can be imported."""
    packages = {
        "langchain_core": "from langchain_core.prompts import PromptTemplate",
        "langgraph": "from langgraph.graph import StateGraph",
        "langchain_google_genai": "from langchain_google_genai import ChatGoogleGenerativeAI",
        "google.generativeai": "import google.generativeai as genai",
        "google.auth": "import google.auth",
        "pydantic": "from pydantic import BaseModel",
        "dotenv": "from dotenv import load_dotenv",
        "click": "import click",
        "tqdm": "from tqdm import tqdm",
        "json": "import json",
        "typing": "from typing import Dict, List, TypedDict",
        "re": "import re",
        "warnings": "import warnings",
    }

    results = {}
    for package_name, import_statement in packages.items():
        try:
            exec(import_statement)
            results[package_name] = (True, "✓")
        except ImportError as e:
            results[package_name] = (False, f"✗ - {str(e)}")
        except Exception as e:
            results[package_name] = (False, f"✗ - {str(e)}")

    return results


def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists and has required variables."""
    env_path = ".env"
    if not os.path.exists(env_path):
        return False, "✗ - .env file not found"

    try:
        from dotenv import load_dotenv

        load_dotenv()

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return False, "✗ - GOOGLE_API_KEY not found in .env"

        if len(google_api_key) < 10:
            return False, "✗ - GOOGLE_API_KEY appears invalid (too short)"

        return True, "✓ - .env file configured"
    except Exception as e:
        return False, f"✗ - Error reading .env: {str(e)}"


def check_google_api_connectivity() -> Tuple[bool, str]:
    """Test Google API connectivity."""
    try:
        from dotenv import load_dotenv
        import google.generativeai as genai

        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            return False, "✗ - No API key available"

        genai.configure(api_key=api_key)

        # Test with a simple model list call
        models = list(genai.list_models())
        if models:
            return True, "✓ - Google API connectivity confirmed"
        else:
            return False, "✗ - No models available"

    except Exception as e:
        return False, f"✗ - API test failed: {str(e)}"


def check_file_permissions() -> Tuple[bool, str]:
    """Check file system permissions for input/output operations."""
    try:
        # Test read permission on dataset
        dataset_path = "dataset/persona_input.json"
        if os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                f.read(100)  # Read first 100 chars

        # Test write permission in current directory
        test_file = "test_write_permission.tmp"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        return True, "✓ - File permissions OK"
    except Exception as e:
        return False, f"✗ - Permission error: {str(e)}"


def main():
    """Run all validation checks."""
    print("🔍 Synthetic Audience Generator - Environment Validation")
    print("=" * 60)

    all_passed = True

    # Python version check
    passed, message = check_python_version()
    print(f"Python Version: {message}")
    all_passed &= passed

    print("\nPackage Imports:")
    package_results = check_package_imports()
    for package, (passed, message) in package_results.items():
        print(f"  {package:20} {message}")
        all_passed &= passed

    print("\nEnvironment Configuration:")
    passed, message = check_env_file()
    print(f"  .env file: {message}")
    all_passed &= passed

    print("\nAPI Connectivity:")
    passed, message = check_google_api_connectivity()
    print(f"  Google API: {message}")
    all_passed &= passed

    print("\nFile System:")
    passed, message = check_file_permissions()
    print(f"  Permissions: {message}")
    all_passed &= passed

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validation checks passed! Environment is ready.")
        sys.exit(0)
    else:
        print("❌ Some validation checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
