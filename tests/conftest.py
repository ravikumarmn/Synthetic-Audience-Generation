"""Test configuration and fixtures for synthetic audience MVP tests."""

import json
import os
import tempfile
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock
import pytest

# Add src directory to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic_audience_mvp import (
    FilterDetails,
    InputPersona,
    RequestData,
    BehavioralContent,
    ProcessingTemplates,
    DemographicAssignment,
    SyntheticProfile,
)


@pytest.fixture
def sample_filter_details():
    """Sample filter details for testing."""
    return FilterDetails(
        user_req_responses=10,
        age_proportions={"18-25": 30, "26-35": 40, "36-45": 30},
        gender_proportions={"Male": 50, "Female": 50},
        ethnicity_proportions={"White": 60, "Hispanic": 25, "Black": 15},
    )


@pytest.fixture
def sample_input_personas():
    """Sample input personas for testing."""
    return [
        InputPersona(
            id=1,
            about="Tech enthusiast who loves coding and innovation",
            goalsAndMotivations=["Learn new technologies", "Build innovative products"],
            frustrations=["Slow development processes", "Outdated tools"],
            needState="Seeking efficient development workflows",
            occasions="During work hours and weekend projects",
        ),
        InputPersona(
            id=2,
            about="Creative professional focused on design and user experience",
            goalsAndMotivations=[
                "Create beautiful interfaces",
                "Improve user satisfaction",
            ],
            frustrations=["Poor design feedback", "Limited creative freedom"],
            needState="Looking for design inspiration and tools",
            occasions="During creative sessions and client meetings",
        ),
    ]


@pytest.fixture
def sample_request_data(sample_filter_details, sample_input_personas):
    """Sample request data for testing."""
    return RequestData(
        request=[
            {
                "filter_details": sample_filter_details.model_dump(),
                "personas": [persona.model_dump() for persona in sample_input_personas],
            }
        ]
    )


@pytest.fixture
def sample_processing_templates():
    """Sample processing templates for testing."""
    return ProcessingTemplates(
        about_templates=[
            "Tech enthusiast who loves coding and innovation",
            "Creative professional focused on design and user experience",
        ],
        goals_templates=[
            "Learn new technologies",
            "Build innovative products",
            "Create beautiful interfaces",
            "Improve user satisfaction",
        ],
        frustrations_templates=[
            "Slow development processes",
            "Outdated tools",
            "Poor design feedback",
            "Limited creative freedom",
        ],
        need_state_templates=[
            "Seeking efficient development workflows",
            "Looking for design inspiration and tools",
        ],
        occasions_templates=[
            "During work hours and weekend projects",
            "During creative sessions and client meetings",
        ],
    )


@pytest.fixture
def sample_demographic_assignments():
    """Sample demographic assignments for testing."""
    return [
        DemographicAssignment(
            age_bucket="18-25", gender="Male", ethnicity="White", profile_index=0
        ),
        DemographicAssignment(
            age_bucket="26-35", gender="Female", ethnicity="Hispanic", profile_index=1
        ),
        DemographicAssignment(
            age_bucket="36-45", gender="Male", ethnicity="Black", profile_index=2
        ),
    ]


@pytest.fixture
def sample_behavioral_content():
    """Sample behavioral content for testing."""
    return BehavioralContent(
        about="Passionate about technology and continuous learning",
        goalsAndMotivations=[
            "Master new programming languages",
            "Contribute to open source projects",
            "Build scalable applications",
        ],
        frustrations=[
            "Legacy code maintenance",
            "Inefficient team processes",
            "Limited learning resources",
        ],
        needState="Actively seeking growth opportunities and technical challenges",
        occasions="During focused work sessions and technical research periods",
    )


@pytest.fixture
def sample_synthetic_profile():
    """Sample synthetic profile for testing."""
    return SyntheticProfile(
        age_bucket="26-35",
        gender="Female",
        ethnicity="Hispanic",
        about="Passionate about technology and continuous learning",
        goalsAndMotivations=[
            "Master new programming languages",
            "Contribute to open source projects",
            "Build scalable applications",
        ],
        frustrations=[
            "Legacy code maintenance",
            "Inefficient team processes",
            "Limited learning resources",
        ],
        needState="Actively seeking growth opportunities and technical challenges",
        occasions="During focused work sessions and technical research periods",
        profile_id=1,
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = Mock()
    mock_response.content = json.dumps(
        {
            "about": "Passionate about technology and continuous learning",
            "goalsAndMotivations": [
                "Master new programming languages",
                "Contribute to open source projects",
                "Build scalable applications",
            ],
            "frustrations": [
                "Legacy code maintenance",
                "Inefficient team processes",
                "Limited learning resources",
            ],
            "needState": "Actively seeking growth opportunities and technical challenges",
            "occasions": "During focused work sessions and technical research periods",
        }
    )
    return mock_response


@pytest.fixture
def temp_input_file(sample_request_data):
    """Create temporary input file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_request_data.model_dump(), f, indent=2)
        temp_file = f.name

    yield temp_file

    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def temp_output_file():
    """Create temporary output file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    yield temp_file

    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def mock_google_llm():
    """Mock Google LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps(
        {
            "about": "Passionate about technology and continuous learning",
            "goalsAndMotivations": [
                "Master new programming languages",
                "Contribute to open source projects",
            ],
            "frustrations": ["Legacy code maintenance", "Inefficient team processes"],
            "needState": "Actively seeking growth opportunities",
            "occasions": "During focused work sessions",
        }
    )
    return mock_llm


@pytest.fixture
def mock_azure_llm():
    """Mock Azure LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps(
        {
            "about": "Creative professional with design focus",
            "goalsAndMotivations": [
                "Create beautiful interfaces",
                "Improve user satisfaction",
            ],
            "frustrations": ["Poor design feedback", "Limited creative freedom"],
            "needState": "Looking for design inspiration",
            "occasions": "During creative sessions",
        }
    )
    return mock_llm


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "test-deployment")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("PARALLEL_BATCH_SIZE", "3")
    monkeypatch.setenv("MAX_WORKERS", "2")
    monkeypatch.setenv("CONCURRENT_REQUESTS", "2")
