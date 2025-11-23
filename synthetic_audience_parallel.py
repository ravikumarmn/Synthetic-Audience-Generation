#!/usr/bin/env python3
import json
import os
import random
from typing import Dict, List, TypedDict, Optional, Any
from collections import Counter
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from prompt import BEHAVIORAL_CONTENT_PROMPT

load_dotenv()


class FilterDetails(BaseModel):
    user_req_responses: int = Field(..., gt=0)
    age_proportions: Dict[str, int]
    gender_proportions: Dict[str, int]
    ethnicity_proportions: Dict[str, int]

    @field_validator("age_proportions", "gender_proportions", "ethnicity_proportions")
    @classmethod
    def validate_proportions_sum_to_100(cls, v):
        if sum(v.values()) != 100:
            raise ValueError(f"Proportions must sum to 100, got {sum(v.values())}")
        return v


class InputPersona(BaseModel):
    id: int
    about: str
    goalsAndMotivations: List[str]
    frustrations: List[str]
    needState: str
    occasions: str


class RequestData(BaseModel):
    request: List[Dict[str, Any]]

    def get_filter_details(self) -> FilterDetails:
        return FilterDetails(**self.request[0]["filter_details"])

    def get_personas(self) -> List[InputPersona]:
        return [InputPersona(**persona) for persona in self.request[0]["personas"]]


class DemographicAssignment(BaseModel):
    age_bucket: str
    gender: str
    ethnicity: str
    profile_index: int


class BehavioralContent(BaseModel):
    about: str
    goalsAndMotivations: List[str]
    frustrations: List[str]
    needState: str
    occasions: str


class SyntheticProfile(BaseModel):
    age_bucket: str
    gender: str
    ethnicity: str
    about: str
    goalsAndMotivations: List[str]
    frustrations: List[str]
    needState: str
    occasions: str
    profile_id: int


class ProcessingTemplates(BaseModel):
    about_templates: List[str]
    goals_templates: List[str]
    frustrations_templates: List[str]
    need_state_templates: List[str]
    occasions_templates: List[str]


class State(TypedDict):
    input_file: str
    output_file: str
    filter_details: Optional[FilterDetails]
    input_personas: Optional[List[InputPersona]]
    processing_templates: Optional[ProcessingTemplates]
    demographic_schedule: Optional[List[DemographicAssignment]]
    batch_1_profiles: Optional[List[SyntheticProfile]]
    batch_2_profiles: Optional[List[SyntheticProfile]]
    batch_3_profiles: Optional[List[SyntheticProfile]]
    batch_4_profiles: Optional[List[SyntheticProfile]]
    all_profiles: Optional[List[SyntheticProfile]]
    generation_stats: Optional[Dict[str, Any]]


class LLMContentGenerator:
    def __init__(self, provider: str = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "google")

        if self.provider == "azure":
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                temperature=0.7,
                max_tokens=1500,
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                max_tokens=1500,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )

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

    def generate_content(self, templates: ProcessingTemplates) -> BehavioralContent:
        prompt = self.prompt_template.format(
            about_examples="\n".join(templates.about_templates[:3]),
            goals_examples="\n".join(templates.goals_templates[:6]),
            frustrations_examples="\n".join(templates.frustrations_templates[:6]),
            need_state_examples="\n".join(templates.need_state_templates[:3]),
            occasions_examples="\n".join(templates.occasions_templates[:3]),
        )

        response = self.llm.invoke(prompt)
        content_data = json.loads(response.content)
        return BehavioralContent(**content_data)


class DistributionCalculator:
    @staticmethod
    def generate_demographic_schedule(
        total: int,
        gender_props: Dict[str, int],
        age_props: Dict[str, int],
        ethnicity_props: Dict[str, int],
    ) -> List[DemographicAssignment]:
        def calculate_exact_counts(
            proportions: Dict[str, int], total: int
        ) -> Dict[str, int]:
            allocations = {}
            remainders = {}

            for category, percentage in proportions.items():
                exact_value = (percentage / 100.0) * total
                allocations[category] = int(exact_value)
                remainders[category] = exact_value - allocations[category]

            total_allocated = sum(allocations.values())
            remaining = total - total_allocated

            if remaining > 0:
                sorted_remainders = sorted(
                    remainders.items(), key=lambda x: x[1], reverse=True
                )
                for i in range(remaining):
                    category = sorted_remainders[i][0]
                    allocations[category] += 1

            return allocations

        gender_counts = calculate_exact_counts(gender_props, total)
        age_counts = calculate_exact_counts(age_props, total)
        ethnicity_counts = calculate_exact_counts(ethnicity_props, total)

        gender_pool = []
        for gender, count in gender_counts.items():
            gender_pool.extend([gender] * count)

        age_pool = []
        for age_bucket, count in age_counts.items():
            age_pool.extend([age_bucket] * count)

        ethnicity_pool = []
        for ethnicity, count in ethnicity_counts.items():
            ethnicity_pool.extend([ethnicity] * count)

        random.shuffle(gender_pool)
        random.shuffle(age_pool)
        random.shuffle(ethnicity_pool)

        schedule = []
        for i in range(total):
            assignment = DemographicAssignment(
                age_bucket=age_pool[i],
                gender=gender_pool[i],
                ethnicity=ethnicity_pool[i],
                profile_index=i,
            )
            schedule.append(assignment)

        return schedule


class PersonaProcessor:
    @staticmethod
    def extract_behavioral_templates(
        personas: List[InputPersona],
    ) -> ProcessingTemplates:
        about_templates = []
        goals_templates = []
        frustrations_templates = []
        need_state_templates = []
        occasions_templates = []

        for persona in personas:
            if persona.about and len(persona.about.strip()) > 10:
                about_templates.append(persona.about.strip())

            for goal in persona.goalsAndMotivations:
                if goal and len(goal.strip()) > 5:
                    goals_templates.append(goal.strip())

            for frustration in persona.frustrations:
                if frustration and len(frustration.strip()) > 5:
                    frustrations_templates.append(frustration.strip())

            if persona.needState and len(persona.needState.strip()) > 5:
                need_state_templates.append(persona.needState.strip())

            if persona.occasions and len(persona.occasions.strip()) > 5:
                occasions_templates.append(persona.occasions.strip())

        return ProcessingTemplates(
            about_templates=list(set(about_templates)),
            goals_templates=list(set(goals_templates)),
            frustrations_templates=list(set(frustrations_templates)),
            need_state_templates=list(set(need_state_templates)),
            occasions_templates=list(set(occasions_templates)),
        )


def load_json_node(state: State) -> State:
    with open(state["input_file"], "r") as f:
        raw_data = json.load(f)

    request_data = RequestData(**raw_data)
    filter_details = request_data.get_filter_details()
    input_personas = request_data.get_personas()

    return {
        **state,
        "filter_details": filter_details,
        "input_personas": input_personas,
    }


def distribution_builder_node(state: State) -> State:
    filter_details = state["filter_details"]

    demographic_schedule = DistributionCalculator.generate_demographic_schedule(
        filter_details.user_req_responses,
        filter_details.gender_proportions,
        filter_details.age_proportions,
        filter_details.ethnicity_proportions,
    )

    return {
        **state,
        "demographic_schedule": demographic_schedule,
    }


def persona_processor_node(state: State) -> State:
    input_personas = state["input_personas"]
    processing_templates = PersonaProcessor.extract_behavioral_templates(input_personas)

    return {
        **state,
        "processing_templates": processing_templates,
    }


def generate_batch_1(state: State) -> State:
    demographic_schedule = state["demographic_schedule"]
    processing_templates = state["processing_templates"]

    batch_size = len(demographic_schedule) // 4
    batch_1 = demographic_schedule[:batch_size]

    generator = LLMContentGenerator()
    profiles = []

    for demographic in batch_1:
        behavioral_content = generator.generate_content(processing_templates)
        profile = SyntheticProfile(
            age_bucket=demographic.age_bucket,
            gender=demographic.gender,
            ethnicity=demographic.ethnicity,
            about=behavioral_content.about,
            goalsAndMotivations=behavioral_content.goalsAndMotivations,
            frustrations=behavioral_content.frustrations,
            needState=behavioral_content.needState,
            occasions=behavioral_content.occasions,
            profile_id=demographic.profile_index + 1,
        )
        profiles.append(profile)

    return {**state, "batch_1_profiles": profiles}


def generate_batch_2(state: State) -> State:
    demographic_schedule = state["demographic_schedule"]
    processing_templates = state["processing_templates"]

    batch_size = len(demographic_schedule) // 4
    batch_2 = demographic_schedule[batch_size : batch_size * 2]

    generator = LLMContentGenerator()
    profiles = []

    for demographic in batch_2:
        behavioral_content = generator.generate_content(processing_templates)
        profile = SyntheticProfile(
            age_bucket=demographic.age_bucket,
            gender=demographic.gender,
            ethnicity=demographic.ethnicity,
            about=behavioral_content.about,
            goalsAndMotivations=behavioral_content.goalsAndMotivations,
            frustrations=behavioral_content.frustrations,
            needState=behavioral_content.needState,
            occasions=behavioral_content.occasions,
            profile_id=demographic.profile_index + 1,
        )
        profiles.append(profile)

    return {**state, "batch_2_profiles": profiles}


def generate_batch_3(state: State) -> State:
    demographic_schedule = state["demographic_schedule"]
    processing_templates = state["processing_templates"]

    batch_size = len(demographic_schedule) // 4
    batch_3 = demographic_schedule[batch_size * 2 : batch_size * 3]

    generator = LLMContentGenerator()
    profiles = []

    for demographic in batch_3:
        behavioral_content = generator.generate_content(processing_templates)
        profile = SyntheticProfile(
            age_bucket=demographic.age_bucket,
            gender=demographic.gender,
            ethnicity=demographic.ethnicity,
            about=behavioral_content.about,
            goalsAndMotivations=behavioral_content.goalsAndMotivations,
            frustrations=behavioral_content.frustrations,
            needState=behavioral_content.needState,
            occasions=behavioral_content.occasions,
            profile_id=demographic.profile_index + 1,
        )
        profiles.append(profile)

    return {**state, "batch_3_profiles": profiles}


def generate_batch_4(state: State) -> State:
    demographic_schedule = state["demographic_schedule"]
    processing_templates = state["processing_templates"]

    batch_size = len(demographic_schedule) // 4
    batch_4 = demographic_schedule[batch_size * 3 :]

    generator = LLMContentGenerator()
    profiles = []

    for demographic in batch_4:
        behavioral_content = generator.generate_content(processing_templates)
        profile = SyntheticProfile(
            age_bucket=demographic.age_bucket,
            gender=demographic.gender,
            ethnicity=demographic.ethnicity,
            about=behavioral_content.about,
            goalsAndMotivations=behavioral_content.goalsAndMotivations,
            frustrations=behavioral_content.frustrations,
            needState=behavioral_content.needState,
            occasions=behavioral_content.occasions,
            profile_id=demographic.profile_index + 1,
        )
        profiles.append(profile)

    return {**state, "batch_4_profiles": profiles}


def aggregator_node(state: State) -> State:
    all_profiles = []
    all_profiles.extend(state["batch_1_profiles"] or [])
    all_profiles.extend(state["batch_2_profiles"] or [])
    all_profiles.extend(state["batch_3_profiles"] or [])
    all_profiles.extend(state["batch_4_profiles"] or [])

    all_profiles.sort(key=lambda x: x.profile_id)

    generation_stats = {
        "total_profiles": len(all_profiles),
        "generation_timestamp": "completed",
    }

    return {
        **state,
        "all_profiles": all_profiles,
        "generation_stats": generation_stats,
    }


def output_writer_node(state: State) -> State:
    all_profiles = state["all_profiles"]
    generation_stats = state["generation_stats"]
    output_file = state.get("output_file", "synthetic_audience.json")

    output_data = {
        "synthetic_audience": [profile.model_dump() for profile in all_profiles],
        "generation_metadata": generation_stats,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    return {**state, "output_written": True}


def create_parallel_workflow() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node("load_json", load_json_node)
    workflow.add_node("distribution_builder", distribution_builder_node)
    workflow.add_node("persona_processor", persona_processor_node)
    workflow.add_node("generate_batch_1", generate_batch_1)
    workflow.add_node("generate_batch_2", generate_batch_2)
    workflow.add_node("generate_batch_3", generate_batch_3)
    workflow.add_node("generate_batch_4", generate_batch_4)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("output_writer", output_writer_node)

    workflow.add_edge(START, "load_json")
    workflow.add_edge("load_json", "distribution_builder")
    workflow.add_edge("distribution_builder", "persona_processor")

    # Parallel generation - all batches run simultaneously
    workflow.add_edge("persona_processor", "generate_batch_1")
    workflow.add_edge("persona_processor", "generate_batch_2")
    workflow.add_edge("persona_processor", "generate_batch_3")
    workflow.add_edge("persona_processor", "generate_batch_4")

    # All batches feed into aggregator
    workflow.add_edge("generate_batch_1", "aggregator")
    workflow.add_edge("generate_batch_2", "aggregator")
    workflow.add_edge("generate_batch_3", "aggregator")
    workflow.add_edge("generate_batch_4", "aggregator")

    workflow.add_edge("aggregator", "output_writer")
    workflow.add_edge("output_writer", END)

    return workflow


class SyntheticAudienceGenerator:
    def __init__(self):
        self.workflow = create_parallel_workflow()
        self.app = self.workflow.compile()

    def process_request(self, input_file: str, output_file: str) -> Dict[str, Any]:
        initial_state = {
            "input_file": input_file,
            "output_file": output_file,
        }

        final_state = self.app.invoke(initial_state)
        return final_state.get("generation_stats", {})


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python synthetic_audience_parallel.py input.json output.json")
        sys.exit(1)

    generator = SyntheticAudienceGenerator()
    stats = generator.process_request(sys.argv[1], sys.argv[2])
    print(f"Generated {stats['total_profiles']} profiles")
