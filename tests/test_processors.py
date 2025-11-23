"""Tests for processors and calculators in synthetic_audience_mvp.py"""

import pytest
from collections import Counter

from synthetic_audience_mvp import (
    DistributionCalculator,
    PersonaProcessor,
    DemographicAssignment,
    ProcessingTemplates,
    InputPersona,
)


class TestDistributionCalculator:
    """Test DistributionCalculator class."""

    def test_generate_demographic_schedule_exact_distribution(self):
        """Test demographic schedule generation with exact proportions."""
        total = 100
        gender_props = {"Male": 50, "Female": 50}
        age_props = {"18-25": 30, "26-35": 40, "36-45": 30}
        ethnicity_props = {"White": 60, "Hispanic": 25, "Black": 15}

        schedule = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )

        assert len(schedule) == total
        assert all(
            isinstance(assignment, DemographicAssignment) for assignment in schedule
        )

        # Check gender distribution
        gender_counts = Counter(assignment.gender for assignment in schedule)
        assert gender_counts["Male"] == 50
        assert gender_counts["Female"] == 50

        # Check age distribution
        age_counts = Counter(assignment.age_bucket for assignment in schedule)
        assert age_counts["18-25"] == 30
        assert age_counts["26-35"] == 40
        assert age_counts["36-45"] == 30

        # Check ethnicity distribution
        ethnicity_counts = Counter(assignment.ethnicity for assignment in schedule)
        assert ethnicity_counts["White"] == 60
        assert ethnicity_counts["Hispanic"] == 25
        assert ethnicity_counts["Black"] == 15

    def test_generate_demographic_schedule_with_remainders(self):
        """Test demographic schedule generation with remainder handling."""
        total = 10
        gender_props = {"Male": 33, "Female": 67}  # 33% = 3.3, 67% = 6.7
        age_props = {"18-25": 100}
        ethnicity_props = {"White": 100}

        schedule = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )

        assert len(schedule) == total

        # Check that remainders are handled correctly
        gender_counts = Counter(assignment.gender for assignment in schedule)
        assert gender_counts["Male"] + gender_counts["Female"] == total
        # With remainder handling, should be 3 Male, 7 Female (or 4 Male, 6 Female)
        assert 3 <= gender_counts["Male"] <= 4
        assert 6 <= gender_counts["Female"] <= 7

    def test_generate_demographic_schedule_small_total(self):
        """Test demographic schedule generation with small total."""
        total = 3
        gender_props = {"Male": 50, "Female": 50}
        age_props = {"18-25": 100}
        ethnicity_props = {"White": 100}

        schedule = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )

        assert len(schedule) == total

        # With total=3 and 50/50 split, should be 1 or 2 of each gender
        gender_counts = Counter(assignment.gender for assignment in schedule)
        assert abs(gender_counts["Male"] - gender_counts["Female"]) <= 1

    def test_generate_demographic_schedule_profile_indices(self):
        """Test that profile indices are correctly assigned."""
        total = 5
        gender_props = {"Male": 100}
        age_props = {"18-25": 100}
        ethnicity_props = {"White": 100}

        schedule = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )

        profile_indices = [assignment.profile_index for assignment in schedule]
        assert sorted(profile_indices) == list(range(total))

    def test_generate_demographic_schedule_randomization(self):
        """Test that demographic assignments are randomized."""
        total = 20
        gender_props = {"Male": 50, "Female": 50}
        age_props = {"18-25": 50, "26-35": 50}
        ethnicity_props = {"White": 100}

        # Generate multiple schedules and check they're different
        schedule1 = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )
        schedule2 = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )

        # Schedules should have same counts but different orders (with high probability)
        genders1 = [a.gender for a in schedule1]
        genders2 = [a.gender for a in schedule2]

        # Same distribution
        assert Counter(genders1) == Counter(genders2)
        # Different order (with very high probability for 20 items)
        assert genders1 != genders2 or len(set(genders1)) == 1  # Unless all same gender

    def test_generate_demographic_schedule_multiple_categories(self):
        """Test with multiple categories in each dimension."""
        total = 12
        gender_props = {"Male": 25, "Female": 50, "Non-binary": 25}
        age_props = {"18-25": 33, "26-35": 33, "36-45": 34}
        ethnicity_props = {"White": 50, "Hispanic": 30, "Black": 20}

        schedule = DistributionCalculator.generate_demographic_schedule(
            total, gender_props, age_props, ethnicity_props
        )

        assert len(schedule) == total

        # Check all categories are represented
        genders = set(assignment.gender for assignment in schedule)
        ages = set(assignment.age_bucket for assignment in schedule)
        ethnicities = set(assignment.ethnicity for assignment in schedule)

        assert genders == {"Male", "Female", "Non-binary"}
        assert ages == {"18-25", "26-35", "36-45"}
        assert ethnicities == {"White", "Hispanic", "Black"}


class TestPersonaProcessor:
    """Test PersonaProcessor class."""

    def test_extract_behavioral_templates_basic(self, sample_input_personas):
        """Test basic template extraction."""
        templates = PersonaProcessor.extract_behavioral_templates(sample_input_personas)

        assert isinstance(templates, ProcessingTemplates)
        assert len(templates.about_templates) == 2
        assert len(templates.goals_templates) == 4
        assert len(templates.frustrations_templates) == 4
        assert len(templates.need_state_templates) == 2
        assert len(templates.occasions_templates) == 2

    def test_extract_behavioral_templates_content(self, sample_input_personas):
        """Test that extracted content matches input."""
        templates = PersonaProcessor.extract_behavioral_templates(sample_input_personas)

        # Check that content from personas is in templates (order not guaranteed due to set conversion)
        assert any(
            "Tech enthusiast" in template for template in templates.about_templates
        )
        assert "Learn new technologies" in templates.goals_templates
        assert "Slow development processes" in templates.frustrations_templates
        assert (
            "Seeking efficient development workflows" in templates.need_state_templates
        )
        assert any(
            "During work hours" in template
            for template in templates.occasions_templates
        ) or any(
            "During creative sessions" in template
            for template in templates.occasions_templates
        )

    def test_extract_behavioral_templates_deduplication(self):
        """Test that duplicate templates are removed."""
        personas = [
            InputPersona(
                id=1,
                about="Same about text",
                goalsAndMotivations=["Same goal", "Different goal"],
                frustrations=["Same frustration"],
                needState="Same need state",
                occasions="Same occasions",
            ),
            InputPersona(
                id=2,
                about="Same about text",  # Duplicate
                goalsAndMotivations=["Same goal", "Another goal"],  # One duplicate
                frustrations=[
                    "Same frustration",
                    "Different frustration",
                ],  # One duplicate
                needState="Same need state",  # Duplicate
                occasions="Same occasions",  # Duplicate
            ),
        ]

        templates = PersonaProcessor.extract_behavioral_templates(personas)

        # Should have deduplicated content
        assert len(templates.about_templates) == 1
        assert (
            len(templates.goals_templates) == 3
        )  # "Same goal", "Different goal", "Another goal"
        assert (
            len(templates.frustrations_templates) == 2
        )  # "Same frustration", "Different frustration"
        assert len(templates.need_state_templates) == 1
        assert len(templates.occasions_templates) == 1

    def test_extract_behavioral_templates_empty_content_filtering(self):
        """Test that empty or short content is filtered out."""
        personas = [
            InputPersona(
                id=1,
                about="",  # Empty
                goalsAndMotivations=[
                    "",
                    "Short",
                    "This is a valid goal",
                ],  # Empty and short
                frustrations=["   ", "Valid frustration"],  # Whitespace only
                needState="Valid need state",
                occasions="Valid occasions",
            ),
            InputPersona(
                id=2,
                about="Short",  # Too short (< 10 chars)
                goalsAndMotivations=["Hi"],  # Too short (< 5 chars)
                frustrations=["This is valid"],
                needState="Hi",  # Too short
                occasions="Valid occasions",
            ),
        ]

        templates = PersonaProcessor.extract_behavioral_templates(personas)

        # Should filter out empty/short content
        assert len(templates.about_templates) == 0  # Both filtered out
        assert "This is a valid goal" in templates.goals_templates
        assert "Short" not in templates.goals_templates
        assert "Valid frustration" in templates.frustrations_templates
        assert "This is valid" in templates.frustrations_templates
        assert len(templates.need_state_templates) == 1  # Only "Valid need state"
        assert len(templates.occasions_templates) == 1  # Only "Valid occasions"

    def test_extract_behavioral_templates_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        personas = [
            InputPersona(
                id=1,
                about="  About with whitespace  ",
                goalsAndMotivations=["  Goal with spaces  "],
                frustrations=["  Frustration with spaces  "],
                needState="  Need state with spaces  ",
                occasions="  Occasions with spaces  ",
            )
        ]

        templates = PersonaProcessor.extract_behavioral_templates(personas)

        # Should have stripped whitespace
        assert templates.about_templates[0] == "About with whitespace"
        assert templates.goals_templates[0] == "Goal with spaces"
        assert templates.frustrations_templates[0] == "Frustration with spaces"
        assert templates.need_state_templates[0] == "Need state with spaces"
        assert templates.occasions_templates[0] == "Occasions with spaces"

    def test_extract_behavioral_templates_empty_input(self):
        """Test extraction with empty persona list."""
        templates = PersonaProcessor.extract_behavioral_templates([])

        assert isinstance(templates, ProcessingTemplates)
        assert len(templates.about_templates) == 0
        assert len(templates.goals_templates) == 0
        assert len(templates.frustrations_templates) == 0
        assert len(templates.need_state_templates) == 0
        assert len(templates.occasions_templates) == 0

    def test_extract_behavioral_templates_mixed_valid_invalid(self):
        """Test extraction with mix of valid and invalid content."""
        personas = [
            InputPersona(
                id=1,
                about="Valid about section with enough content",
                goalsAndMotivations=["Valid goal", "", "Another valid goal"],
                frustrations=["", "Valid frustration"],
                needState="Valid need state",
                occasions="Valid occasions",
            )
        ]

        templates = PersonaProcessor.extract_behavioral_templates(personas)

        assert len(templates.about_templates) == 1
        assert len(templates.goals_templates) == 2  # Two valid goals
        assert len(templates.frustrations_templates) == 1  # One valid frustration
        assert len(templates.need_state_templates) == 1
        assert len(templates.occasions_templates) == 1
