#!/bin/bash

MY_BENCHMARK_FILE="./xls/examples/proc_network_opt_ir_benchmark.sh"

die () {
  echo "ERROR: $1"
  exit 1
}

# Run the benchmark
${MY_BENCHMARK_FILE} || die "Failed in ${MY_BENCHMARK_FILE}"

echo "PASS"
