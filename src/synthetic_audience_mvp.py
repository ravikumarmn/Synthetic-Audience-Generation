#!/usr/bin/env python3
import json
import os
import random
import asyncio
import argparse
import time
from typing import Dict, List, TypedDict, Optional, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from prompt import BEHAVIORAL_CONTENT_PROMPT

load_dotenv()


def clean_json_response(response_content: str) -> str:
    """Clean LLM response content to extract valid JSON."""
    content = response_content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```

    if content.endswith("```"):
        content = content[:-3]  # Remove closing ```

    # Find JSON object boundaries
    content = content.strip()

    # Look for the first { and last }
    start_idx = content.find("{")
    end_idx = content.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        content = content[start_idx : end_idx + 1]

    return content.strip()


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
    all_profiles: Optional[List[SyntheticProfile]]
    generation_stats: Optional[Dict[str, Any]]
    use_async: Optional[bool]
    batch_size: Optional[int]
    max_workers: Optional[int]
    concurrent_requests: Optional[int]


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
                max_tokens=4000,  # Increased from 1500 to handle longer responses
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

        cleaned_content = clean_json_response(response.content)
        content_data = json.loads(cleaned_content)
        return BehavioralContent(**content_data)


class AsyncLLMContentGenerator:
    """Async wrapper for LLM content generation with ThreadPoolExecutor."""

    def __init__(self, provider: str = None, max_workers: int = None):
        self.generator = LLMContentGenerator(provider)
        self.max_workers = max_workers or int(os.getenv("MAX_WORKERS", "3"))

    async def generate_content_async(
        self, templates: ProcessingTemplates
    ) -> BehavioralContent:
        """Generate content asynchronously using ThreadPoolExecutor."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            result = await loop.run_in_executor(
                executor, self.generator.generate_content, templates
            )
        return result


class ParallelBatchProcessor:
    """Handles parallel batch processing with semaphore control."""

    def __init__(
        self,
        provider: str = None,
        batch_size: int = None,
        max_workers: int = None,
        concurrent_requests: int = None,
    ):
        self.provider = provider
        self.batch_size = batch_size or int(os.getenv("PARALLEL_BATCH_SIZE", "5"))
        self.max_workers = max_workers or int(os.getenv("MAX_WORKERS", "3"))
        self.concurrent_requests = concurrent_requests or int(
            os.getenv("CONCURRENT_REQUESTS", "3")
        )

    async def process_profiles_parallel(
        self, demographics: List[DemographicAssignment], templates: ProcessingTemplates
    ) -> List[SyntheticProfile]:
        """Process all profiles in parallel batches with semaphore control."""

        # Create semaphore to control concurrent requests
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        async_generator = AsyncLLMContentGenerator(self.provider, self.max_workers)

        async def generate_single_profile(
            demographic: DemographicAssignment,
        ) -> SyntheticProfile:
            """Generate a single profile with semaphore control."""
            async with semaphore:
                behavioral_content = await async_generator.generate_content_async(
                    templates
                )
                return SyntheticProfile(
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

        # Create tasks for all profiles
        tasks = [generate_single_profile(demo) for demo in demographics]

        # Process in batches to avoid overwhelming the system
        all_profiles = []
        for i in range(0, len(tasks), self.batch_size):
            batch_tasks = tasks[i : i + self.batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle any exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error generating profile: {result}")
                else:
                    all_profiles.append(result)

            # Small delay between batches to be respectful to API limits
            if i + self.batch_size < len(tasks):
                await asyncio.sleep(0.1)

        return all_profiles


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


def generate_profiles_sequential(state: State) -> State:
    """Sequential profile generation (fallback method)."""
    demographic_schedule = state["demographic_schedule"]
    processing_templates = state["processing_templates"]
    provider = state.get("provider", "google")

    generator = LLMContentGenerator(provider)
    profiles = []

    print(f"Generating {len(demographic_schedule)} profiles sequentially...")
    for i, demographic in enumerate(demographic_schedule, 1):
        print(f"  Processing profile {i}/{len(demographic_schedule)}")
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

    return {**state, "all_profiles": profiles}


def generate_profiles_async(state: State) -> State:
    """Async parallel profile generation (recommended method)."""
    demographic_schedule = state["demographic_schedule"]
    processing_templates = state["processing_templates"]

    # Get configuration from state or environment
    provider = state.get("provider", "google")
    batch_size = state.get("batch_size", int(os.getenv("PARALLEL_BATCH_SIZE", "5")))
    max_workers = state.get("max_workers", int(os.getenv("MAX_WORKERS", "3")))
    concurrent_requests = state.get(
        "concurrent_requests", int(os.getenv("CONCURRENT_REQUESTS", "3"))
    )

    print(f"Generating {len(demographic_schedule)} profiles with async processing...")
    print(
        f"Configuration: batch_size={batch_size}, max_workers={max_workers}, concurrent_requests={concurrent_requests}"
    )

    # Create and run the parallel processor
    processor = ParallelBatchProcessor(
        provider=provider,
        batch_size=batch_size,
        max_workers=max_workers,
        concurrent_requests=concurrent_requests,
    )

    # Run async processing
    start_time = time.time()
    profiles = asyncio.run(
        processor.process_profiles_parallel(demographic_schedule, processing_templates)
    )
    end_time = time.time()

    print(f"Async processing completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {len(profiles)} profiles successfully")

    return {**state, "all_profiles": profiles}


def finalize_profiles_node(state: State) -> State:
    """Finalize profiles and generate statistics."""
    all_profiles = state["all_profiles"]

    # Sort profiles by ID to ensure consistent ordering
    all_profiles.sort(key=lambda x: x.profile_id)

    generation_stats = {
        "total_profiles": len(all_profiles),
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_mode": "async" if state.get("use_async", True) else "sequential",
        "batch_size": state.get("batch_size", "N/A"),
        "max_workers": state.get("max_workers", "N/A"),
        "concurrent_requests": state.get("concurrent_requests", "N/A"),
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


def create_async_workflow() -> StateGraph:
    """Create workflow with async parallel processing."""
    workflow = StateGraph(State)

    workflow.add_node("load_json", load_json_node)
    workflow.add_node("distribution_builder", distribution_builder_node)
    workflow.add_node("persona_processor", persona_processor_node)
    workflow.add_node("generate_profiles_async", generate_profiles_async)
    workflow.add_node("finalize_profiles", finalize_profiles_node)
    workflow.add_node("output_writer", output_writer_node)

    workflow.add_edge(START, "load_json")
    workflow.add_edge("load_json", "distribution_builder")
    workflow.add_edge("distribution_builder", "persona_processor")
    workflow.add_edge("persona_processor", "generate_profiles_async")
    workflow.add_edge("generate_profiles_async", "finalize_profiles")
    workflow.add_edge("finalize_profiles", "output_writer")
    workflow.add_edge("output_writer", END)

    return workflow


def create_sequential_workflow() -> StateGraph:
    """Create workflow with sequential processing (fallback)."""
    workflow = StateGraph(State)

    workflow.add_node("load_json", load_json_node)
    workflow.add_node("distribution_builder", distribution_builder_node)
    workflow.add_node("persona_processor", persona_processor_node)
    workflow.add_node("generate_profiles_sequential", generate_profiles_sequential)
    workflow.add_node("finalize_profiles", finalize_profiles_node)
    workflow.add_node("output_writer", output_writer_node)

    workflow.add_edge(START, "load_json")
    workflow.add_edge("load_json", "distribution_builder")
    workflow.add_edge("distribution_builder", "persona_processor")
    workflow.add_edge("persona_processor", "generate_profiles_sequential")
    workflow.add_edge("generate_profiles_sequential", "finalize_profiles")
    workflow.add_edge("finalize_profiles", "output_writer")
    workflow.add_edge("output_writer", END)

    return workflow


# Legacy function name for backward compatibility
def create_parallel_workflow() -> StateGraph:
    """Legacy function - use create_async_workflow() instead."""
    return create_async_workflow()


class SyntheticAudienceGenerator:
    """Enhanced Synthetic Audience Generator with async processing capabilities."""

    def __init__(
        self,
        use_async: bool = True,
        provider: str = None,
        batch_size: int = None,
        max_workers: int = None,
        concurrent_requests: int = None,
    ):
        """
        Initialize the generator with configuration options.

        Args:
            use_async: Whether to use async processing (default: True)
            provider: LLM provider ('google' or 'azure')
            batch_size: Number of profiles to process in each batch
            max_workers: Number of worker threads
            concurrent_requests: Number of concurrent API requests
        """
        self.use_async = use_async
        self.provider = provider or os.getenv("LLM_PROVIDER", "google")
        self.batch_size = batch_size or int(os.getenv("PARALLEL_BATCH_SIZE", "5"))
        self.max_workers = max_workers or int(os.getenv("MAX_WORKERS", "3"))
        self.concurrent_requests = concurrent_requests or int(
            os.getenv("CONCURRENT_REQUESTS", "3")
        )

        # Create appropriate workflow
        if self.use_async:
            self.workflow = create_async_workflow()
        else:
            self.workflow = create_sequential_workflow()

        self.app = self.workflow.compile()

    def process_request(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process the synthetic audience generation request."""
        initial_state = {
            "input_file": input_file,
            "output_file": output_file,
            "use_async": self.use_async,
            "provider": self.provider,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "concurrent_requests": self.concurrent_requests,
        }

        print(f"🚀 Starting Synthetic Audience Generation")
        print(f"Mode: {'Async Parallel' if self.use_async else 'Sequential'}")
        print(f"Provider: {self.provider}")
        if self.use_async:
            print(
                f"Config: batch_size={self.batch_size}, max_workers={self.max_workers}, concurrent_requests={self.concurrent_requests}"
            )
        print("-" * 60)

        start_time = time.time()
        final_state = self.app.invoke(initial_state)
        end_time = time.time()

        stats = final_state.get("generation_stats", {})
        stats["total_processing_time"] = f"{end_time - start_time:.2f} seconds"

        print(f"\n✅ Generation completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Generated {stats.get('total_profiles', 0)} profiles")

        return stats

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the current workflow configuration."""
        return {
            "mode": "async" if self.use_async else "sequential",
            "provider": self.provider,
            "batch_size": self.batch_size if self.use_async else "N/A",
            "max_workers": self.max_workers if self.use_async else "N/A",
            "concurrent_requests": (
                self.concurrent_requests if self.use_async else "N/A"
            ),
        }


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic audience profiles with async processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with async processing (recommended)
  python src/synthetic_audience_mvp.py input.json output.json
  
  # Use sequential processing
  python src/synthetic_audience_mvp.py input.json output.json --sequential
  
  # Custom async configuration for 100 profiles
  python src/synthetic_audience_mvp.py input.json output.json --batch-size 10 --max-workers 5 --concurrent-requests 5
  
  # Use Azure OpenAI provider
  python src/synthetic_audience_mvp.py input.json output.json --provider azure
  
  # Performance test mode
  python src/synthetic_audience_mvp.py --show-config
        """,
    )

    # Required arguments (unless showing help/tips)
    parser.add_argument(
        "input_file", nargs="?", help="Input JSON file with personas and demographics"
    )
    parser.add_argument(
        "output_file", nargs="?", help="Output JSON file for generated profiles"
    )

    # Processing mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        default=True,
        help="Use async parallel processing (default)",
    )
    mode_group.add_argument(
        "--sequential",
        dest="use_async",
        action="store_false",
        help="Use sequential processing (slower but more stable)",
    )

    # LLM Provider
    parser.add_argument(
        "--provider",
        choices=["google", "azure"],
        default="google",
        help="LLM provider to use (default: google)",
    )

    # Async configuration
    async_group = parser.add_argument_group(
        "async configuration", "Options for async parallel processing"
    )
    async_group.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of profiles to process in each batch (default: 5)",
    )
    async_group.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of worker threads (default: 3)",
    )
    async_group.add_argument(
        "--concurrent-requests",
        type=int,
        default=3,
        help="Number of concurrent API requests (default: 3)",
    )

    # Utility options
    parser.add_argument(
        "--show-config", action="store_true", help="Show configuration and exit"
    )
    parser.add_argument(
        "--performance-tips",
        action="store_true",
        help="Show performance optimization tips",
    )

    return parser


def show_performance_tips():
    """Display performance optimization tips."""
    print(
        """
🚀 Performance Optimization Tips:

📊 Dataset Size Recommendations:
  • Small (< 10 profiles):    --sequential (less overhead)
  • Medium (10-50 profiles):  --batch-size 5 --max-workers 3
  • Large (50-100 profiles):  --batch-size 10 --max-workers 5
  • XLarge (100+ profiles):   --batch-size 15 --max-workers 8

⚡ Speed vs Stability:
  • Fastest:     --concurrent-requests 8 --max-workers 8
  • Balanced:    --concurrent-requests 5 --max-workers 5 (recommended)
  • Stable:      --concurrent-requests 3 --max-workers 3
  • Most Stable: --sequential

🔧 API Rate Limit Management:
  • If you hit rate limits: Reduce --concurrent-requests
  • For Google Gemini free tier: --concurrent-requests 2
  • For Azure OpenAI: --concurrent-requests 10+ (depending on quota)

💡 Environment Variables (alternative to CLI):
  export PARALLEL_BATCH_SIZE=10
  export MAX_WORKERS=5
  export CONCURRENT_REQUESTS=5
  export LLM_PROVIDER=azure
    """
    )


if __name__ == "__main__":
    parser = create_cli_parser()
    args = parser.parse_args()

    # Show performance tips if requested
    if args.performance_tips:
        show_performance_tips()
        exit(0)

    # Validate required arguments
    if not args.input_file or not args.output_file:
        if not args.show_config:
            parser.error(
                "input_file and output_file are required unless using --show-config or --performance-tips"
            )

    # Create generator with configuration
    generator = SyntheticAudienceGenerator(
        use_async=args.use_async,
        provider=args.provider,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        concurrent_requests=args.concurrent_requests,
    )

    # Show configuration if requested
    if args.show_config:
        config = generator.get_workflow_info()
        print("\n🔧 Current Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("\n💡 Use --performance-tips for optimization guidance")
        exit(0)

    # Process the request
    try:
        stats = generator.process_request(args.input_file, args.output_file)

        print("\n📈 Final Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        print("\n💡 Try using --sequential mode if async processing fails")
        exit(1)
