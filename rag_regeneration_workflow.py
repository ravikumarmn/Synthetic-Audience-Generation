#!/usr/bin/env python3
"""
RAG-Enhanced Workflow with Regeneration Logic

This workflow implements:
1. Parallel LLM generation
2. RAG duplicate detection
3. Automatic regeneration when duplicates are found
4. Write to JSON only when content is unique
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, END

from simple_rag_detector import SimpleRAGDetector, SimilarityResult

logger = logging.getLogger(__name__)


class RAGRegenerationState(TypedDict, total=False):
    """Enhanced state for RAG workflow with regeneration."""

    # Input/Output
    input_file: str
    output_file: str

    # Processing data
    filter_details: Dict[str, Any]
    personas: List[Dict[str, Any]]
    templates: Dict[str, Any]
    demographic_schedule: List[Dict[str, Any]]

    # Generation tracking
    current_batch_index: int
    total_batches: int
    current_profiles_batch: List[Dict[str, Any]]

    # RAG system
    rag_detector: SimpleRAGDetector
    similarity_threshold: float
    max_regeneration_attempts: int

    # Results tracking
    generated_profiles: List[Dict[str, Any]]
    duplicate_profiles: List[Dict[str, Any]]
    regeneration_stats: Dict[str, Any]

    # Final output
    synthetic_profiles: List[Dict[str, Any]]
    generation_stats: Dict[str, Any]
    processing_errors: List[str]


@dataclass
class RegenerationResult:
    """Result of content generation with regeneration logic."""

    profile: Dict[str, Any]
    was_duplicate: bool
    regeneration_attempts: int
    similarity_results: Dict[str, SimilarityResult]
    final_content: Dict[str, Any]


class RAGRegenerationWorkflow:
    """Workflow that regenerates content when duplicates are found."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_regeneration_attempts: int = 3,
        batch_size: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_regeneration_attempts = max_regeneration_attempts
        self.batch_size = batch_size

        # Initialize RAG detector
        self.rag_detector = SimpleRAGDetector(similarity_threshold=similarity_threshold)

        logger.info(f"Initialized RAG Regeneration Workflow:")
        logger.info(f"  - Similarity threshold: {similarity_threshold}")
        logger.info(f"  - Max regeneration attempts: {max_regeneration_attempts}")
        logger.info(f"  - Batch size: {batch_size}")


def load_json_node(state: RAGRegenerationState) -> RAGRegenerationState:
    """Load and validate input JSON file."""
    logger.info(f"📁 Loading JSON from: {state['input_file']}")

    try:
        with open(state["input_file"], "r") as f:
            data = json.load(f)

        state["filter_details"] = data.get("filterDetails", {})
        state["personas"] = data.get("personas", [])

        # Initialize RAG detector
        state["rag_detector"] = SimpleRAGDetector(
            similarity_threshold=state.get("similarity_threshold", 0.85)
        )

        # Initialize tracking
        state["generated_profiles"] = []
        state["duplicate_profiles"] = []
        state["regeneration_stats"] = {
            "total_generated": 0,
            "total_duplicates": 0,
            "total_regenerations": 0,
            "successful_regenerations": 0,
            "failed_regenerations": 0,
        }

        logger.info(f"✅ Loaded {len(state['personas'])} personas")

    except Exception as e:
        error_msg = f"Failed to load JSON: {str(e)}"
        logger.error(error_msg)
        state["processing_errors"] = state.get("processing_errors", []) + [error_msg]

    return state


def distribution_builder_node(state: RAGRegenerationState) -> RAGRegenerationState:
    """Build demographic distribution schedule."""
    logger.info("📊 Building demographic distribution schedule...")

    try:
        filter_details = state["filter_details"]
        total_profiles = filter_details.get("user_req_responses", 10)

        # Create demographic schedule (simplified for demo)
        demographic_schedule = []
        for i in range(total_profiles):
            demographic_schedule.append(
                {
                    "profile_id": i + 1,
                    "age_bucket": "25-34",  # Simplified
                    "gender": "Male" if i % 2 == 0 else "Female",
                    "ethnicity": "White",  # Simplified
                }
            )

        state["demographic_schedule"] = demographic_schedule
        state["total_batches"] = (
            len(demographic_schedule) + state.get("batch_size", 5) - 1
        ) // state.get("batch_size", 5)
        state["current_batch_index"] = 0

        logger.info(
            f"✅ Created schedule for {len(demographic_schedule)} profiles in {state['total_batches']} batches"
        )

    except Exception as e:
        error_msg = f"Failed to build distribution: {str(e)}"
        logger.error(error_msg)
        state["processing_errors"] = state.get("processing_errors", []) + [error_msg]

    return state


def persona_processor_node(state: RAGRegenerationState) -> RAGRegenerationState:
    """Process input personas and extract templates."""
    logger.info("👥 Processing input personas...")

    try:
        personas = state["personas"]

        # Extract templates from personas (simplified)
        templates = {
            "about_templates": [p.get("about", "") for p in personas if p.get("about")],
            "goals_templates": [
                goal for p in personas for goal in p.get("goalsAndMotivations", [])
            ],
            "frustrations_templates": [
                frust for p in personas for frust in p.get("frustrations", [])
            ],
            "need_state_templates": [
                p.get("needState", "") for p in personas if p.get("needState")
            ],
            "occasions_templates": [
                p.get("occasions", "") for p in personas if p.get("occasions")
            ],
        }

        state["templates"] = templates

        logger.info(f"✅ Extracted templates:")
        for key, values in templates.items():
            logger.info(f"  - {key}: {len(values)} items")

    except Exception as e:
        error_msg = f"Failed to process personas: {str(e)}"
        logger.error(error_msg)
        state["processing_errors"] = state.get("processing_errors", []) + [error_msg]

    return state


def prepare_batch_node(state: RAGRegenerationState) -> RAGRegenerationState:
    """Prepare the next batch of profiles for generation."""
    logger.info(
        f"🔄 Preparing batch {state['current_batch_index'] + 1}/{state['total_batches']}"
    )

    try:
        batch_size = state.get("batch_size", 5)
        start_idx = state["current_batch_index"] * batch_size
        end_idx = min(start_idx + batch_size, len(state["demographic_schedule"]))

        current_batch = state["demographic_schedule"][start_idx:end_idx]
        state["current_profiles_batch"] = current_batch

        logger.info(f"✅ Prepared batch with {len(current_batch)} profiles")

    except Exception as e:
        error_msg = f"Failed to prepare batch: {str(e)}"
        logger.error(error_msg)
        state["processing_errors"] = state.get("processing_errors", []) + [error_msg]

    return state


async def generate_single_profile_with_regeneration(
    profile_demo: Dict[str, Any],
    templates: Dict[str, Any],
    rag_detector: SimpleRAGDetector,
    max_attempts: int = 3,
) -> RegenerationResult:
    """Generate a single profile with regeneration logic."""

    profile_id = profile_demo["profile_id"]
    regeneration_attempts = 0

    while regeneration_attempts < max_attempts:
        try:
            # Simulate LLM content generation (replace with actual LLM call)
            generated_content = {
                "about": f"Generated about content for profile {profile_id} (attempt {regeneration_attempts + 1})",
                "goalsAndMotivations": [
                    f"Goal A for profile {profile_id}",
                    f"Goal B for profile {profile_id}",
                ],
                "frustrations": [
                    f"Frustration A for profile {profile_id}",
                    f"Frustration B for profile {profile_id}",
                ],
                "needState": f"Need state for profile {profile_id}",
                "occasions": f"Occasions for profile {profile_id}",
            }

            # Check for duplicates using RAG
            final_content, similarity_results = rag_detector.upsert_behavioral_content(
                generated_content, profile_id=profile_id
            )

            # Check if any section was marked as similar (duplicate)
            has_duplicates = any(
                result.is_similar for result in similarity_results.values()
            )

            if not has_duplicates:
                # No duplicates found - success!
                complete_profile = {
                    **profile_demo,  # Demographics
                    **final_content,  # Behavioral content
                }

                return RegenerationResult(
                    profile=complete_profile,
                    was_duplicate=False,
                    regeneration_attempts=regeneration_attempts,
                    similarity_results=similarity_results,
                    final_content=final_content,
                )
            else:
                # Duplicates found - need to regenerate
                regeneration_attempts += 1
                duplicate_sections = [
                    section
                    for section, result in similarity_results.items()
                    if result.is_similar
                ]

                logger.info(
                    f"🔄 Profile {profile_id} has duplicates in {duplicate_sections}. "
                    f"Regenerating (attempt {regeneration_attempts}/{max_attempts})"
                )

                if regeneration_attempts >= max_attempts:
                    # Max attempts reached - use the content anyway but mark as duplicate
                    complete_profile = {**profile_demo, **final_content}

                    return RegenerationResult(
                        profile=complete_profile,
                        was_duplicate=True,
                        regeneration_attempts=regeneration_attempts,
                        similarity_results=similarity_results,
                        final_content=final_content,
                    )

                # Add some variation for next attempt (in real implementation, modify LLM prompt)
                await asyncio.sleep(0.1)  # Small delay before retry

        except Exception as e:
            logger.error(f"Error generating profile {profile_id}: {e}")
            regeneration_attempts += 1

            if regeneration_attempts >= max_attempts:
                # Return error profile
                error_profile = {
                    **profile_demo,
                    "about": f"Error generating content: {str(e)}",
                    "goalsAndMotivations": [],
                    "frustrations": [],
                    "needState": "Error state",
                    "occasions": "Error occasions",
                }

                return RegenerationResult(
                    profile=error_profile,
                    was_duplicate=False,
                    regeneration_attempts=regeneration_attempts,
                    similarity_results={},
                    final_content={},
                )

    # Should not reach here, but just in case
    raise Exception(
        f"Failed to generate profile {profile_id} after {max_attempts} attempts"
    )


async def parallel_rag_generation_node(
    state: RAGRegenerationState,
) -> RAGRegenerationState:
    """Generate profiles in parallel with RAG duplicate detection and regeneration."""
    logger.info("⚡ Starting parallel RAG generation with regeneration...")

    try:
        current_batch = state["current_profiles_batch"]
        templates = state["templates"]
        rag_detector = state["rag_detector"]
        max_attempts = state.get("max_regeneration_attempts", 3)

        # Generate all profiles in the batch concurrently
        tasks = [
            generate_single_profile_with_regeneration(
                profile_demo, templates, rag_detector, max_attempts
            )
            for profile_demo in current_batch
        ]

        # Wait for all generations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_generated = []
        batch_duplicates = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Generation failed: {result}")
                state["regeneration_stats"]["failed_regenerations"] += 1
                continue

            if result.was_duplicate:
                batch_duplicates.append(result.profile)
                state["regeneration_stats"]["total_duplicates"] += 1
                state["regeneration_stats"]["failed_regenerations"] += 1
            else:
                batch_generated.append(result.profile)
                state["regeneration_stats"]["total_generated"] += 1

                if result.regeneration_attempts > 0:
                    state["regeneration_stats"]["successful_regenerations"] += 1

            state["regeneration_stats"][
                "total_regenerations"
            ] += result.regeneration_attempts

        # Add to overall results
        state["generated_profiles"].extend(batch_generated)
        state["duplicate_profiles"].extend(batch_duplicates)

        logger.info(f"✅ Batch completed:")
        logger.info(f"  - Generated: {len(batch_generated)} profiles")
        logger.info(f"  - Duplicates: {len(batch_duplicates)} profiles")

    except Exception as e:
        error_msg = f"Failed parallel generation: {str(e)}"
        logger.error(error_msg)
        state["processing_errors"] = state.get("processing_errors", []) + [error_msg]

    return state


def should_continue_generation(state: RAGRegenerationState) -> str:
    """Determine if more batches need processing."""
    current_batch = state.get("current_batch_index", 0)
    total_batches = state.get("total_batches", 0)

    if current_batch < total_batches - 1:
        return "prepare_batch"  # Process next batch
    else:
        return "write_unique_profiles"  # All batches done, write results


def increment_batch_node(state: RAGRegenerationState) -> RAGRegenerationState:
    """Increment batch counter for next iteration."""
    state["current_batch_index"] = state.get("current_batch_index", 0) + 1
    logger.info(f"📈 Moving to batch {state['current_batch_index'] + 1}")
    return state


def write_unique_profiles_node(state: RAGRegenerationState) -> RAGRegenerationState:
    """Write only unique (non-duplicate) profiles to JSON file."""
    logger.info("💾 Writing unique profiles to JSON...")

    try:
        output_file = state["output_file"]
        generated_profiles = state["generated_profiles"]

        # Prepare output data
        output_data = {
            "synthetic_profiles": generated_profiles,
            "generation_stats": {
                "total_requested": len(state.get("demographic_schedule", [])),
                "total_generated": len(generated_profiles),
                "total_duplicates_rejected": len(state.get("duplicate_profiles", [])),
                "regeneration_stats": state.get("regeneration_stats", {}),
                "rag_stats": (
                    state["rag_detector"].get_stats()
                    if state.get("rag_detector")
                    else {}
                ),
            },
        }

        # Write to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        state["synthetic_profiles"] = generated_profiles
        state["generation_stats"] = output_data["generation_stats"]

        logger.info(
            f"✅ Successfully wrote {len(generated_profiles)} unique profiles to {output_file}"
        )
        logger.info(
            f"📊 Rejected {len(state.get('duplicate_profiles', []))} duplicate profiles"
        )

    except Exception as e:
        error_msg = f"Failed to write output: {str(e)}"
        logger.error(error_msg)
        state["processing_errors"] = state.get("processing_errors", []) + [error_msg]

    return state


def create_rag_regeneration_workflow(
    similarity_threshold: float = 0.85,
    max_regeneration_attempts: int = 3,
    batch_size: int = 5,
) -> StateGraph:
    """Create RAG workflow with regeneration logic."""
    logger.info("🚀 Creating RAG Regeneration Workflow...")

    workflow = StateGraph(RAGRegenerationState)

    # Add nodes
    workflow.add_node("load_json", load_json_node)
    workflow.add_node("distribution_builder", distribution_builder_node)
    workflow.add_node("persona_processor", persona_processor_node)
    workflow.add_node("prepare_batch", prepare_batch_node)
    workflow.add_node("parallel_rag_generation", parallel_rag_generation_node)
    workflow.add_node("increment_batch", increment_batch_node)
    workflow.add_node("write_unique_profiles", write_unique_profiles_node)

    # Define edges
    workflow.set_entry_point("load_json")
    workflow.add_edge("load_json", "distribution_builder")
    workflow.add_edge("distribution_builder", "persona_processor")
    workflow.add_edge("persona_processor", "prepare_batch")
    workflow.add_edge("prepare_batch", "parallel_rag_generation")

    # Conditional edge for batch processing
    workflow.add_conditional_edges(
        "parallel_rag_generation",
        should_continue_generation,
        {
            "prepare_batch": "increment_batch",  # More batches to process
            "write_unique_profiles": "write_unique_profiles",  # All done
        },
    )

    workflow.add_edge("increment_batch", "prepare_batch")  # Loop back for next batch
    workflow.add_edge("write_unique_profiles", END)

    return workflow


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create workflow
        workflow = create_rag_regeneration_workflow(
            similarity_threshold=0.8, max_regeneration_attempts=3, batch_size=3
        )

        app = workflow.compile()

        # Test state
        initial_state = {
            "input_file": "test_input.json",
            "output_file": "test_output.json",
            "similarity_threshold": 0.8,
            "max_regeneration_attempts": 3,
            "batch_size": 3,
        }

        print("🧪 Testing RAG Regeneration Workflow...")

        # Create test input file
        test_input = {
            "filterDetails": {"user_req_responses": 5},
            "personas": [
                {
                    "about": "Test persona about",
                    "goalsAndMotivations": ["Test goal 1", "Test goal 2"],
                    "frustrations": ["Test frustration 1"],
                    "needState": "Test need state",
                    "occasions": "Test occasions",
                }
            ],
        }

        with open("test_input.json", "w") as f:
            json.dump(test_input, f)

        try:
            # Run workflow
            final_state = app.invoke(initial_state)

            print("✅ Workflow completed!")
            print(f"📊 Generation Stats: {final_state.get('generation_stats', {})}")

        except Exception as e:
            print(f"❌ Workflow failed: {e}")

    asyncio.run(main())
