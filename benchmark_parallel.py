#!/usr/bin/env python3
"""
Benchmark Script: Serial vs Parallel Optimization

This script provides empirical evidence of parallelization speedup by:
1. Running both serial and parallel optimizations with identical configs
2. Measuring execution time and throughput
3. Verifying results are identical
4. Generating performance report

Usage:
    python benchmark_parallel.py --quick  # Fast benchmark
    python benchmark_parallel.py          # Full benchmark
"""

import sys
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any
import multiprocessing as mp

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import typer
from loguru import logger
import pandas as pd

app = typer.Typer(help="Benchmark parallel vs serial optimization")


def run_serial_optimization(window_days: int, timeframe: str, test_windows: int, quick_mode: bool) -> Dict[str, Any]:
    """Run serial optimization and measure performance."""
    from optimize_portfolio_comprehensive import ComprehensiveOptimizer

    logger.info("\n" + "="*80)
    logger.info("RUNNING SERIAL OPTIMIZATION")
    logger.info("="*80)

    start_time = time.time()

    optimizer = ComprehensiveOptimizer(
        window_days=window_days,
        timeframe=timeframe,
        test_windows=test_windows,
        quick_mode=quick_mode
    )

    # Suppress detailed logging
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    result = optimizer.optimize(output_dir="benchmark_results_serial")

    duration = time.time() - start_time

    logger.info(f"\n✓ Serial optimization complete")
    logger.info(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"  Configurations tested: {len(result['all_results'])}")
    logger.info(f"  Best outperformance: {result['best_config']['test_avg_outperformance']:.2%}")

    return {
        'duration': duration,
        'configs_tested': len(result['all_results']),
        'best_outperformance': result['best_config']['test_avg_outperformance'],
        'best_config_id': result['best_config']['config_id'],
        'throughput': len(result['all_results']) / duration  # configs per second
    }


def run_parallel_optimization(window_days: int, timeframe: str, test_windows: int,
                              quick_mode: bool, workers: int) -> Dict[str, Any]:
    """Run parallel optimization and measure performance."""
    from optimize_portfolio_parallel import ParallelOptimizer

    logger.info("\n" + "="*80)
    logger.info(f"RUNNING PARALLEL OPTIMIZATION ({workers} workers)")
    logger.info("="*80)

    start_time = time.time()

    optimizer = ParallelOptimizer(
        window_days=window_days,
        timeframe=timeframe,
        test_windows=test_windows,
        quick_mode=quick_mode,
        workers=workers
    )

    # Suppress detailed logging
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    result = optimizer.optimize(output_dir="benchmark_results_parallel")

    duration = time.time() - start_time

    logger.info(f"\n✓ Parallel optimization complete")
    logger.info(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"  Configurations tested: {len(result['all_results'])}")
    logger.info(f"  Best outperformance: {result['best_config']['test_avg_outperformance']:.2%}")
    logger.info(f"  Workers used: {result['workers_used']}")

    return {
        'duration': duration,
        'configs_tested': len(result['all_results']),
        'best_outperformance': result['best_config']['test_avg_outperformance'],
        'best_config_id': result['best_config']['config_id'],
        'workers': result['workers_used'],
        'throughput': len(result['all_results']) / duration  # configs per second
    }


def verify_results_identical(serial_results: Dict, parallel_results: Dict) -> bool:
    """Verify that serial and parallel produce identical results."""
    checks = {
        'configs_tested': serial_results['configs_tested'] == parallel_results['configs_tested'],
        'best_config_id': serial_results['best_config_id'] == parallel_results['best_config_id'],
        'best_outperformance_close': abs(serial_results['best_outperformance'] -
                                         parallel_results['best_outperformance']) < 1e-6
    }

    all_passed = all(checks.values())

    logger.info("\n" + "="*80)
    logger.info("RESULT VERIFICATION")
    logger.info("="*80)

    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {check_name}: {status}")

    if all_passed:
        logger.success("\n✅ All verification checks passed - results are identical")
    else:
        logger.error("\n❌ Verification failed - results differ between serial and parallel")

    return all_passed


def generate_benchmark_report(serial: Dict, parallel: Dict, output_file: str = "BENCHMARK_REPORT.txt"):
    """Generate comprehensive benchmark report."""

    speedup = serial['duration'] / parallel['duration']
    efficiency = speedup / parallel['workers'] * 100

    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("PARALLELIZATION BENCHMARK REPORT\n")
        f.write("Serial vs Parallel Optimization Performance Comparison\n")
        f.write("="*100 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CPU Cores: {mp.cpu_count()}\n")
        f.write(f"Workers Used: {parallel['workers']}\n\n")

        # Executive Summary
        f.write("="*100 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("="*100 + "\n\n")

        f.write(f"🚀 SPEEDUP: {speedup:.2f}x faster with parallel execution\n")
        f.write(f"⚡ EFFICIENCY: {efficiency:.1f}% parallel efficiency\n")
        f.write(f"⏱️  TIME SAVED: {serial['duration'] - parallel['duration']:.1f} seconds ({(serial['duration'] - parallel['duration'])/60:.2f} minutes)\n\n")

        if speedup >= 10:
            f.write("✅ EXCELLENT: Achieved >10x speedup - highly effective parallelization\n")
        elif speedup >= 5:
            f.write("✓ GOOD: Achieved 5-10x speedup - effective parallelization\n")
        elif speedup >= 2:
            f.write("⚠ MODERATE: Achieved 2-5x speedup - some parallelization benefit\n")
        else:
            f.write("❌ POOR: <2x speedup - parallelization not effective\n")

        f.write("\n")

        # Detailed Performance Metrics
        f.write("="*100 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*100 + "\n\n")

        f.write("SERIAL EXECUTION:\n")
        f.write(f"  Duration:           {serial['duration']:>10.2f} seconds ({serial['duration']/60:>6.2f} minutes)\n")
        f.write(f"  Configs Tested:     {serial['configs_tested']:>10}\n")
        f.write(f"  Throughput:         {serial['throughput']:>10.2f} configs/second\n")
        f.write(f"  Best Outperformance:{serial['best_outperformance']:>10.2%}\n\n")

        f.write("PARALLEL EXECUTION:\n")
        f.write(f"  Duration:           {parallel['duration']:>10.2f} seconds ({parallel['duration']/60:>6.2f} minutes)\n")
        f.write(f"  Configs Tested:     {parallel['configs_tested']:>10}\n")
        f.write(f"  Throughput:         {parallel['throughput']:>10.2f} configs/second\n")
        f.write(f"  Best Outperformance:{parallel['best_outperformance']:>10.2%}\n")
        f.write(f"  Workers:            {parallel['workers']:>10}\n\n")

        f.write("COMPARISON:\n")
        f.write(f"  Speedup:            {speedup:>10.2f}x\n")
        f.write(f"  Time Saved:         {serial['duration'] - parallel['duration']:>10.2f} seconds\n")
        f.write(f"  Throughput Gain:    {parallel['throughput'] / serial['throughput']:>10.2f}x\n")
        f.write(f"  Parallel Efficiency:{efficiency:>10.1f}%\n\n")

        # Theoretical Analysis
        f.write("="*100 + "\n")
        f.write("THEORETICAL ANALYSIS\n")
        f.write("="*100 + "\n\n")

        theoretical_speedup = parallel['workers']
        amdahl_overhead = 1 - (speedup / theoretical_speedup)

        f.write(f"Theoretical Maximum Speedup:  {theoretical_speedup:.1f}x ({parallel['workers']} workers)\n")
        f.write(f"Actual Speedup:               {speedup:.2f}x\n")
        f.write(f"Efficiency:                   {efficiency:.1f}% of theoretical maximum\n\n")

        f.write(f"Amdahl's Law Analysis:\n")
        f.write(f"  Serial fraction (overhead):  {amdahl_overhead:.1%}\n")
        f.write(f"  Parallel fraction:           {1 - amdahl_overhead:.1%}\n\n")

        if efficiency >= 90:
            f.write("  Assessment: EXCELLENT - Near-linear scaling\n")
        elif efficiency >= 70:
            f.write("  Assessment: GOOD - Effective parallelization with acceptable overhead\n")
        elif efficiency >= 50:
            f.write("  Assessment: MODERATE - Significant overhead, but still beneficial\n")
        else:
            f.write("  Assessment: POOR - High overhead limits parallelization benefits\n")

        # Scalability Projection
        f.write("\n\n" + "="*100 + "\n")
        f.write("SCALABILITY PROJECTIONS\n")
        f.write("="*100 + "\n\n")

        f.write("Estimated execution time with different worker counts:\n")
        f.write(f"  {1:>3} worker:  {serial['duration']:>8.1f} seconds ({serial['duration']/60:>6.2f} minutes)\n")

        for workers in [2, 4, 8, 16, 32]:
            if workers <= mp.cpu_count():
                # Use observed efficiency to project
                projected_speedup = workers * (efficiency / 100)
                projected_time = serial['duration'] / projected_speedup
                f.write(f"  {workers:>3} workers: {projected_time:>8.1f} seconds ({projected_time/60:>6.2f} minutes) - {projected_speedup:>5.2f}x speedup\n")

        # Evidence and Proof
        f.write("\n\n" + "="*100 + "\n")
        f.write("EVIDENCE & PROOF\n")
        f.write("="*100 + "\n\n")

        f.write("CORRECTNESS VERIFICATION:\n")
        if serial['best_config_id'] == parallel['best_config_id']:
            f.write("  ✅ Both methods identified the same best configuration\n")
        else:
            f.write("  ⚠️  Different configurations selected (may indicate race condition)\n")

        if abs(serial['best_outperformance'] - parallel['best_outperformance']) < 1e-6:
            f.write("  ✅ Performance metrics match exactly\n")
        else:
            f.write("  ⚠️  Performance metrics differ slightly\n")

        if serial['configs_tested'] == parallel['configs_tested']:
            f.write("  ✅ Same number of configurations tested\n")
        else:
            f.write("  ⚠️  Different number of configurations (possible filtering difference)\n")

        f.write("\nPERFORMANCE EVIDENCE:\n")
        f.write(f"  Serial duration:   {serial['duration']:.2f}s (measured)\n")
        f.write(f"  Parallel duration: {parallel['duration']:.2f}s (measured)\n")
        f.write(f"  Speedup ratio:     {speedup:.2f}x (calculated: serial/parallel)\n")
        f.write(f"  Time saved:        {serial['duration'] - parallel['duration']:.2f}s (measured difference)\n")

        # Recommendations
        f.write("\n\n" + "="*100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*100 + "\n\n")

        f.write("1. DEPLOYMENT:\n")
        if speedup >= 5:
            f.write("   ✅ Use parallel version for all production optimizations\n")
            f.write(f"   ✅ Optimal worker count: {parallel['workers']} (tested configuration)\n")
        elif speedup >= 2:
            f.write("   ✓ Use parallel version for large optimizations\n")
            f.write("   ⚠ Serial may be acceptable for quick tests\n")
        else:
            f.write("   ⚠ Overhead too high - evaluate if parallelization is worth it\n")

        f.write("\n2. OPTIMIZATION:\n")
        if efficiency < 70:
            f.write("   - Reduce inter-process communication\n")
            f.write("   - Use larger chunk sizes\n")
            f.write("   - Consider shared memory for data\n")
        else:
            f.write("   ✓ Current implementation is well-optimized\n")

        f.write("\n3. SCALING:\n")
        if efficiency >= 80:
            f.write("   ✅ Excellent scaling - can benefit from more cores\n")
        elif efficiency >= 60:
            f.write("   ✓ Good scaling - moderate benefit from more cores\n")
        else:
            f.write("   ⚠ Poor scaling - limited benefit from additional cores\n")

        f.write("\n\n" + "="*100 + "\n")
        f.write("END OF BENCHMARK REPORT\n")
        f.write("="*100 + "\n")

    logger.info(f"\n✓ Benchmark report saved to: {output_file}")


@app.command()
def benchmark(
    quick: bool = typer.Option(True, "--quick", "-q", help="Quick benchmark with reduced grid"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of parallel workers (default: auto)")
):
    """
    Benchmark serial vs parallel optimization performance.

    This provides empirical evidence of parallelization speedup by running
    identical optimizations in both modes and comparing performance.

    Example:
        python benchmark_parallel.py --quick
        python benchmark_parallel.py --workers 8
    """
    logger.info("\n" + "="*80)
    logger.info("PARALLELIZATION BENCHMARK")
    logger.info("="*80)
    logger.info(f"\nSystem Information:")
    logger.info(f"  CPU Cores: {mp.cpu_count()}")
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)
    logger.info(f"  Workers to test: {workers}")
    logger.info(f"  Quick mode: {quick}")

    # Configuration
    window_days = 90 if quick else 365
    timeframe = "1d" if quick else "1h"
    test_windows = 2 if quick else 3

    logger.info(f"\nBenchmark Configuration:")
    logger.info(f"  Window: {window_days} days")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Test windows: {test_windows}")

    try:
        # Run serial optimization
        serial_results = run_serial_optimization(window_days, timeframe, test_windows, quick)

        # Run parallel optimization
        parallel_results = run_parallel_optimization(window_days, timeframe, test_windows, quick, workers)

        # Verify results match
        results_match = verify_results_identical(serial_results, parallel_results)

        # Calculate speedup
        speedup = serial_results['duration'] / parallel_results['duration']
        time_saved = serial_results['duration'] - parallel_results['duration']

        # Print summary
        logger.info("\n" + "="*80)
        logger.success("BENCHMARK COMPLETE")
        logger.info("="*80)
        logger.info(f"\n🚀 SPEEDUP: {speedup:.2f}x")
        logger.info(f"⏱️  Serial time:   {serial_results['duration']:.2f}s ({serial_results['duration']/60:.2f} min)")
        logger.info(f"⏱️  Parallel time: {parallel_results['duration']:.2f}s ({parallel_results['duration']/60:.2f} min)")
        logger.info(f"💰 Time saved:   {time_saved:.2f}s ({time_saved/60:.2f} min)")
        logger.info(f"✓ Results match: {results_match}")

        # Generate report
        generate_benchmark_report(serial_results, parallel_results)

        logger.info("\n" + "="*80)
        logger.info("📊 Detailed report: BENCHMARK_REPORT.txt")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n❌ Benchmark failed: {e}")
        logger.exception("Full traceback:")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
