"""Tests for Pydantic models in synthetic_audience_mvp.py"""

import pytest
from pydantic import ValidationError

from synthetic_audience_mvp import (
    FilterDetails,
    InputPersona,
    RequestData,
    BehavioralContent,
    ProcessingTemplates,
    DemographicAssignment,
    SyntheticProfile,
)


class TestFilterDetails:
    """Test FilterDetails model validation."""

    def test_valid_filter_details(self, sample_filter_details):
        """Test valid filter details creation."""
        assert sample_filter_details.user_req_responses == 10
        assert sample_filter_details.age_proportions["18-25"] == 30
        assert sample_filter_details.gender_proportions["Male"] == 50
        assert sample_filter_details.ethnicity_proportions["White"] == 60

    def test_proportions_sum_to_100(self):
        """Test that proportions must sum to 100."""
        with pytest.raises(ValidationError) as exc_info:
            FilterDetails(
                user_req_responses=10,
                age_proportions={"18-25": 30, "26-35": 40, "36-45": 20},  # Sum = 90
                gender_proportions={"Male": 50, "Female": 50},
                ethnicity_proportions={"White": 60, "Hispanic": 25, "Black": 15},
            )
        assert "Proportions must sum to 100" in str(exc_info.value)

    def test_zero_responses_invalid(self):
        """Test that zero responses is invalid."""
        with pytest.raises(ValidationError):
            FilterDetails(
                user_req_responses=0,
                age_proportions={"18-25": 100},
                gender_proportions={"Male": 100},
                ethnicity_proportions={"White": 100},
            )

    def test_negative_responses_invalid(self):
        """Test that negative responses is invalid."""
        with pytest.raises(ValidationError):
            FilterDetails(
                user_req_responses=-5,
                age_proportions={"18-25": 100},
                gender_proportions={"Male": 100},
                ethnicity_proportions={"White": 100},
            )


class TestInputPersona:
    """Test InputPersona model validation."""

    def test_valid_input_persona(self, sample_input_personas):
        """Test valid input persona creation."""
        persona = sample_input_personas[0]
        assert persona.id == 1
        assert "Tech enthusiast" in persona.about
        assert len(persona.goalsAndMotivations) == 2
        assert len(persona.frustrations) == 2
        assert persona.needState == "Seeking efficient development workflows"
        assert "work hours" in persona.occasions

    def test_empty_lists_allowed(self):
        """Test that empty lists are allowed for goals and frustrations."""
        persona = InputPersona(
            id=1,
            about="Test persona",
            goalsAndMotivations=[],
            frustrations=[],
            needState="Test need state",
            occasions="Test occasions",
        )
        assert persona.goalsAndMotivations == []
        assert persona.frustrations == []

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            InputPersona(
                id=1,
                about="Test persona",
                # Missing required fields
            )


class TestRequestData:
    """Test RequestData model validation."""

    def test_valid_request_data(self, sample_request_data):
        """Test valid request data creation."""
        assert len(sample_request_data.request) == 1
        filter_details = sample_request_data.get_filter_details()
        assert filter_details.user_req_responses == 10

        personas = sample_request_data.get_personas()
        assert len(personas) == 2
        assert personas[0].id == 1

    def test_get_filter_details(self, sample_request_data):
        """Test filter details extraction."""
        filter_details = sample_request_data.get_filter_details()
        assert isinstance(filter_details, FilterDetails)
        assert filter_details.user_req_responses == 10

    def test_get_personas(self, sample_request_data):
        """Test personas extraction."""
        personas = sample_request_data.get_personas()
        assert len(personas) == 2
        assert all(isinstance(p, InputPersona) for p in personas)


class TestBehavioralContent:
    """Test BehavioralContent model validation."""

    def test_valid_behavioral_content(self, sample_behavioral_content):
        """Test valid behavioral content creation."""
        assert "technology" in sample_behavioral_content.about
        assert len(sample_behavioral_content.goalsAndMotivations) == 3
        assert len(sample_behavioral_content.frustrations) == 3
        assert "growth opportunities" in sample_behavioral_content.needState
        assert "work sessions" in sample_behavioral_content.occasions

    def test_empty_lists_allowed(self):
        """Test that empty lists are allowed."""
        content = BehavioralContent(
            about="Test about",
            goalsAndMotivations=[],
            frustrations=[],
            needState="Test need state",
            occasions="Test occasions",
        )
        assert content.goalsAndMotivations == []
        assert content.frustrations == []


class TestProcessingTemplates:
    """Test ProcessingTemplates model validation."""

    def test_valid_processing_templates(self, sample_processing_templates):
        """Test valid processing templates creation."""
        assert len(sample_processing_templates.about_templates) == 2
        assert len(sample_processing_templates.goals_templates) == 4
        assert len(sample_processing_templates.frustrations_templates) == 4
        assert len(sample_processing_templates.need_state_templates) == 2
        assert len(sample_processing_templates.occasions_templates) == 2

    def test_empty_templates_allowed(self):
        """Test that empty template lists are allowed."""
        templates = ProcessingTemplates(
            about_templates=[],
            goals_templates=[],
            frustrations_templates=[],
            need_state_templates=[],
            occasions_templates=[],
        )
        assert len(templates.about_templates) == 0
        assert len(templates.goals_templates) == 0
        assert len(templates.frustrations_templates) == 0
        assert len(templates.need_state_templates) == 0
        assert len(templates.occasions_templates) == 0


class TestDemographicAssignment:
    """Test DemographicAssignment model validation."""

    def test_valid_demographic_assignment(self, sample_demographic_assignments):
        """Test valid demographic assignment creation."""
        assignment = sample_demographic_assignments[0]
        assert assignment.age_bucket == "18-25"
        assert assignment.gender == "Male"
        assert assignment.ethnicity == "White"
        assert assignment.profile_index == 0

    def test_negative_profile_index_allowed(self):
        """Test that negative profile index is allowed (edge case)."""
        assignment = DemographicAssignment(
            age_bucket="18-25", gender="Male", ethnicity="White", profile_index=-1
        )
        assert assignment.profile_index == -1


class TestSyntheticProfile:
    """Test SyntheticProfile model validation."""

    def test_valid_synthetic_profile(self, sample_synthetic_profile):
        """Test valid synthetic profile creation."""
        assert sample_synthetic_profile.age_bucket == "26-35"
        assert sample_synthetic_profile.gender == "Female"
        assert sample_synthetic_profile.ethnicity == "Hispanic"
        assert "technology" in sample_synthetic_profile.about
        assert len(sample_synthetic_profile.goalsAndMotivations) == 3
        assert len(sample_synthetic_profile.frustrations) == 3
        assert sample_synthetic_profile.profile_id == 1

    def test_profile_serialization(self, sample_synthetic_profile):
        """Test profile can be serialized to dict."""
        profile_dict = sample_synthetic_profile.model_dump()
        assert isinstance(profile_dict, dict)
        assert profile_dict["profile_id"] == 1
        assert profile_dict["age_bucket"] == "26-35"
        assert isinstance(profile_dict["goalsAndMotivations"], list)

    def test_profile_from_dict(self, sample_synthetic_profile):
        """Test profile can be created from dict."""
        profile_dict = sample_synthetic_profile.model_dump()
        new_profile = SyntheticProfile(**profile_dict)
        assert new_profile.profile_id == sample_synthetic_profile.profile_id
        assert new_profile.about == sample_synthetic_profile.about
