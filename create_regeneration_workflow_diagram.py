#!/usr/bin/env python3
"""
Create RAG Regeneration Workflow Visualization

This script creates a visual representation of the RAG workflow with
regeneration logic for duplicate handling.
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END


class RAGRegenerationState(TypedDict, total=False):
    """State for RAG regeneration workflow."""

    input_file: str
    output_file: str
    current_batch_index: int
    total_batches: int
    generated_profiles: list
    duplicate_profiles: list
    regeneration_stats: Dict[str, Any]


def load_json_node(state: RAGRegenerationState):
    """Load and validate input JSON file."""
    print("📁 Loading JSON input...")
    return state


def distribution_builder_node(state: RAGRegenerationState):
    """Build demographic distribution schedule."""
    print("📊 Building demographic distribution...")
    return state


def persona_processor_node(state: RAGRegenerationState):
    """Process input personas and extract templates."""
    print("👥 Processing input personas...")
    return state


def prepare_batch_node(state: RAGRegenerationState):
    """Prepare next batch of profiles for generation."""
    print("🔄 Preparing batch for generation...")
    return state


def parallel_rag_generation_node(state: RAGRegenerationState):
    """Generate profiles with RAG duplicate detection and regeneration."""
    print("⚡ Parallel generation with RAG regeneration...")
    return state


def increment_batch_node(state: RAGRegenerationState):
    """Increment batch counter."""
    print("📈 Moving to next batch...")
    return state


def write_unique_profiles_node(state: RAGRegenerationState):
    """Write only unique profiles to JSON."""
    print("💾 Writing unique profiles to JSON...")
    return state


def should_continue_generation(state: RAGRegenerationState) -> str:
    """Determine if more batches need processing."""
    current_batch = state.get("current_batch_index", 0)
    total_batches = state.get("total_batches", 1)

    if current_batch < total_batches - 1:
        return "prepare_batch"
    else:
        return "write_unique_profiles"


def create_rag_regeneration_workflow() -> StateGraph:
    """Create RAG workflow with regeneration logic."""
    print("🚀 Creating RAG Regeneration Workflow...")

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

    # Conditional edge for batch processing with regeneration loop
    workflow.add_conditional_edges(
        "parallel_rag_generation",
        should_continue_generation,
        {
            "prepare_batch": "increment_batch",  # More batches to process
            "write_unique_profiles": "write_unique_profiles",  # All done
        },
    )

    workflow.add_edge("increment_batch", "prepare_batch")  # Loop back
    workflow.add_edge("write_unique_profiles", END)

    return workflow


def visualize_regeneration_workflow(save_path: str = "workflow_rag_regeneration.png"):
    """Visualize the RAG regeneration workflow."""
    try:
        # Create workflow
        workflow = create_rag_regeneration_workflow()
        app = workflow.compile()

        print(f"📊 Creating RAG Regeneration Workflow visualization...")

        # Generate PNG
        try:
            png_data = app.get_graph().draw_mermaid_png()

            with open(save_path, "wb") as f:
                f.write(png_data)

            print(f"✅ Regeneration workflow diagram saved to: {save_path}")

            # Try to display if in Jupyter
            try:
                from IPython.display import Image, display

                display(Image(png_data))
                print("📱 Workflow displayed in notebook")
            except ImportError:
                pass

            return True

        except Exception as e:
            print(f"❌ Error generating PNG: {e}")

            # Fallback: show Mermaid code
            try:
                mermaid_code = app.get_graph().draw_mermaid()
                print(f"\n📋 Mermaid code for RAG Regeneration Workflow:")
                print("=" * 60)
                print(mermaid_code)
                print("=" * 60)
                print(f"💡 Copy this code to https://mermaid.live to visualize")
                return True
            except Exception as e2:
                print(f"❌ Error generating Mermaid code: {e2}")
                return False

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False


def main():
    """Main function to create regeneration workflow visualization."""
    print("🎨 RAG Regeneration Workflow Visualization Generator")
    print("=" * 60)

    success = visualize_regeneration_workflow("workflow_rag_regeneration.png")

    if success:
        print(f"\n🎉 RAG regeneration workflow visualization completed!")
        print(f"📊 Diagram: workflow_rag_regeneration.png")

        print("\n📋 Regeneration Workflow Features:")
        print("• Parallel LLM generation in batches")
        print("• RAG duplicate detection per profile")
        print("• Automatic regeneration when duplicates found")
        print("• Write only unique profiles to JSON")
        print("• Configurable regeneration attempts")
        print("• Comprehensive statistics tracking")

        print("\n🔄 Regeneration Logic:")
        print("1. Generate profile content")
        print("2. Check for duplicates using RAG")
        print("3. If duplicate found → regenerate with variations")
        print("4. If unique → add to output")
        print("5. If max attempts reached → mark as duplicate")
        print("6. Write only unique profiles to JSON file")

    else:
        print("\n❌ Failed to create regeneration workflow visualization")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
