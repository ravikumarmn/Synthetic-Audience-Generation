#!/usr/bin/env python3
"""
Example usage of Synthetic Audience Generator with Azure OpenAI.

This script demonstrates how to use Azure OpenAI instead of Google Gemini
for generating synthetic audience profiles.
"""

import os
from synthetic_audience_mvp import SyntheticAudienceGenerator


def main():
    """Demonstrate Azure OpenAI integration."""

    # Input and output files
    input_file = "dataset/small_demo_input.json"
    output_file = "results/azure_output.json"

    print("🚀 Synthetic Audience Generator - Azure OpenAI Demo")
    print("=" * 60)

    # Configure Azure OpenAI
    print("\n🔧 Configuring Azure OpenAI...")

    # Set Azure OpenAI configuration
    os.environ["LLM_PROVIDER"] = "azure"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "your_azure_openai_api_key_here"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "your_deployment_name"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"

    # Optional: Configure parallel processing
    os.environ["PARALLEL_BATCH_SIZE"] = "3"
    os.environ["MAX_WORKERS"] = "2"
    os.environ["CONCURRENT_REQUESTS"] = "2"

    print("✅ Azure OpenAI configuration set")
    print(f"   Endpoint: {os.environ['AZURE_OPENAI_ENDPOINT']}")
    print(f"   Deployment: {os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}")
    print(f"   API Version: {os.environ['AZURE_OPENAI_API_VERSION']}")

    try:
        # Initialize generator with parallel processing
        print("\n📊 Initializing Azure OpenAI generator...")
        generator = SyntheticAudienceGenerator(use_parallel=True)

        # Process the request
        print(f"\n🎯 Generating synthetic audience profiles...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")

        result = generator.process_request(input_file, output_file)

        print(f"\n✅ Generation completed successfully!")
        print(f"📄 Output saved to: {output_file}")
        print(f"📊 Generated {result.get('total_profiles', 'N/A')} profiles")

    except ValueError as e:
        if "Azure OpenAI configuration missing" in str(e):
            print(f"\n❌ Configuration Error: {str(e)}")
            print("\n💡 To fix this:")
            print("1. Update the Azure OpenAI configuration in this script")
            print("2. Or set the environment variables in your .env file:")
            print("   - AZURE_OPENAI_ENDPOINT")
            print("   - AZURE_OPENAI_API_KEY")
            print("   - AZURE_OPENAI_DEPLOYMENT_NAME")
            print("   - AZURE_OPENAI_API_VERSION")
        else:
            print(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")

    print("\n💡 Usage Tips:")
    print(
        "• Use CLI: python synthetic_audience_mvp.py --provider azure -i input.json -o output.json"
    )
    print("• Set environment variables in .env file for persistent configuration")
    print("• Azure OpenAI typically has higher rate limits than Google Gemini")
    print("• Supports both GPT-3.5-turbo and GPT-4 deployments")


if __name__ == "__main__":
    main()
