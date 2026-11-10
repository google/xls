#!/usr/bin/env bash

MY_BENCHMARK_FILE="./xls/examples/proc_network_opt_ir_benchmark.sh"

die () {
  echo "ERROR: $1"
  exit 1
}

# Run the benchmark
${MY_BENCHMARK_FILE} || die "Failed in ${MY_BENCHMARK_FILE}"

# Verify fractional square-micron area is preserved without integer truncation.
BENCHMARK_MAIN="./xls/dev_tools/benchmark_main"
FRACTIONAL_AREA_IR="${TEST_TMPDIR}/fractional_area.ir"
cat > "${FRACTIONAL_AREA_IR}" <<EOF
package fractional_area

top fn inv(x: bits[1]) -> bits[1] {
  ret result: bits[1] = not(x)
}
EOF

AREA_OUTPUT=$("${BENCHMARK_MAIN}" --area_model=asap7 --run_evaluators=false "${FRACTIONAL_AREA_IR}") || die "Failed running benchmark_main on ${FRACTIONAL_AREA_IR}"
echo "${AREA_OUTPUT}" | grep -F "Total area: 0.0437 um2" || die "Expected fractional Total area 0.0437 um2, got: ${AREA_OUTPUT}"

echo "PASS"
