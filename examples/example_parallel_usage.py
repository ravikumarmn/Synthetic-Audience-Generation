#!/usr/bin/env python3
"""
Example usage of Synthetic Audience Generator with parallel processing.

This script demonstrates how to use the parallel processing capabilities
to generate synthetic audience profiles much faster than sequential processing.
"""

import os
import time
from synthetic_audience_mvp import SyntheticAudienceGenerator


def main():
    """Demonstrate parallel vs sequential processing."""

    # Input and output files
    input_file = "dataset/small_demo_input.json"
    output_file_parallel = "results/parallel_output.json"
    output_file_sequential = "results/sequential_output.json"

    print("🚀 Synthetic Audience Generator - Parallel Processing Demo")
    print("=" * 60)

    # Test parallel processing
    print("\n📊 Testing PARALLEL processing...")
    start_time = time.time()

    try:
        # Configure parallel processing
        os.environ["PARALLEL_BATCH_SIZE"] = "3"  # Process 3 profiles at once
        os.environ["MAX_WORKERS"] = "2"  # Use 2 worker threads
        os.environ["CONCURRENT_REQUESTS"] = "2"  # Allow 2 concurrent API calls

        generator_parallel = SyntheticAudienceGenerator(use_parallel=True)
        result_parallel = generator_parallel.process_request(
            input_file, output_file_parallel
        )

        parallel_time = time.time() - start_time
        print(f"✅ Parallel processing completed in {parallel_time:.2f} seconds")
        print(f"📄 Output saved to: {output_file_parallel}")

    except Exception as e:
        print(f"❌ Parallel processing failed: {str(e)}")
        return

    # Test sequential processing for comparison
    print("\n📊 Testing SEQUENTIAL processing...")
    start_time = time.time()

    try:
        generator_sequential = SyntheticAudienceGenerator(use_parallel=False)
        result_sequential = generator_sequential.process_request(
            input_file, output_file_sequential
        )

        sequential_time = time.time() - start_time
        print(f"✅ Sequential processing completed in {sequential_time:.2f} seconds")
        print(f"📄 Output saved to: {output_file_sequential}")

    except Exception as e:
        print(f"❌ Sequential processing failed: {str(e)}")
        return

    # Compare results
    print("\n📈 Performance Comparison:")
    print(f"   Parallel:   {parallel_time:.2f} seconds")
    print(f"   Sequential: {sequential_time:.2f} seconds")

    if parallel_time < sequential_time:
        speedup = sequential_time / parallel_time
        print(f"   🎯 Speedup: {speedup:.2f}x faster with parallel processing!")
    else:
        print("   ⚠️  Sequential was faster (possibly due to small dataset or overhead)")

    print("\n💡 Tips for optimal parallel performance:")
    print("   • Use batch sizes of 3-10 for best results")
    print("   • Increase concurrent requests for larger datasets")
    print("   • Monitor API rate limits to avoid throttling")
    print("   • Parallel processing works best with 10+ profiles")


if __name__ == "__main__":
    main()
