#!/usr/bin/env python3
"""
Example script demonstrating LangGraph workflow visualization
for the Synthetic Audience Generator.

This script shows how to:
1. Create the workflow graph
2. Generate Mermaid diagram code
3. Save the workflow as a PNG image
4. Display the workflow (in Jupyter environments)
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from synthetic_audience_mvp import SyntheticAudienceGenerator


def main():
    """Demonstrate workflow visualization capabilities."""
    print("🔄 Initializing Synthetic Audience Generator...")

    try:
        # Initialize the generator
        generator = SyntheticAudienceGenerator()
        print("✅ Generator initialized successfully")

        # 1. Show Mermaid code
        print("\n" + "=" * 60)
        print("📊 LANGGRAPH WORKFLOW MERMAID CODE")
        print("=" * 60)

        mermaid_code = generator.get_workflow_mermaid_code()
        if mermaid_code:
            print(mermaid_code)
            print("\n💡 You can copy this code to https://mermaid.live/ to view online")
        else:
            print("❌ Failed to generate Mermaid code")

        # 2. Save workflow diagram as PNG
        print("\n" + "=" * 60)
        print("💾 SAVING WORKFLOW DIAGRAM")
        print("=" * 60)

        output_path = "workflow_diagram.png"
        result = generator.visualize_workflow(
            save_path=output_path,
            display_image=False,  # Don't try to display in terminal
        )

        if result:
            print(f"✅ Workflow diagram saved to: {output_path}")
            print(f"📁 File size: {len(result)} bytes")
        else:
            print("❌ Failed to save workflow diagram")

        # 3. Show workflow structure info
        print("\n" + "=" * 60)
        print("🏗️  WORKFLOW STRUCTURE INFO")
        print("=" * 60)

        # Get workflow graph info
        graph = generator.app.get_graph()
        nodes = list(graph.nodes.keys())
        edges = list(graph.edges)

        print(f"📋 Total Nodes: {len(nodes)}")
        print("🔗 Workflow Nodes:")
        for i, node in enumerate(nodes, 1):
            print(f"   {i}. {node}")

        print(f"\n🔀 Total Edges: {len(edges)}")
        print("➡️  Workflow Flow:")
        for edge in edges:
            print(f"   {edge[0]} → {edge[1]}")

        print("\n" + "=" * 60)
        print("✅ VISUALIZATION DEMO COMPLETED")
        print("=" * 60)
        print("📖 Usage Examples:")
        print("   • View Mermaid code: python synthetic_audience_mvp.py --show-mermaid")
        print(
            "   • Save diagram: python synthetic_audience_mvp.py --save-graph workflow.png"
        )
        print("   • Display in Jupyter: python synthetic_audience_mvp.py --visualize")
        print(
            "   • Combined: python synthetic_audience_mvp.py --show-mermaid --save-graph diagram.png"
        )

    except Exception as e:
        print(f"❌ Error during visualization demo: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
