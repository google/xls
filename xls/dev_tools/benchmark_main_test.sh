#!/usr/bin/env bash

MY_BENCHMARK_FILE="./xls/examples/proc_network_opt_ir_benchmark.sh"

die () {
  echo "ERROR: $1"
  exit 1
}

# Run the benchmark and verify that block evaluation elaborates the top
# container block (ProcName) rather than blocks()[0].
BENCHMARK_LOG="${TEST_TMPDIR}/proc_network_benchmark.log"
${MY_BENCHMARK_FILE} --vmodule=benchmark_main=1 --stderrthreshold=0 2> "${BENCHMARK_LOG}" || die "Failed in ${MY_BENCHMARK_FILE}"
grep -F "Evaluating block: ProcName" "${BENCHMARK_LOG}" || die "Expected benchmark_main to evaluate top block ProcName, got: $(cat "${BENCHMARK_LOG}")"

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
