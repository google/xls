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

# Verify shared opt_flags (such as --skip_passes) are accepted by benchmark_main.
SKIP_OUTPUT=$("${BENCHMARK_MAIN}" --area_model=asap7 --skip_passes=dce --run_evaluators=false "${FRACTIONAL_AREA_IR}") || die "Failed running benchmark_main with --skip_passes"
echo "${SKIP_OUTPUT}" | grep -E "Dynamic pass count: [0-9]+" || die "Expected Dynamic pass count in output, got: ${SKIP_OUTPUT}"

# Verify --opt_options_proto selects top on multi-function DSLX-generated IR and
# conflicting top values between --opt_options_proto and --codegen_options_proto
# are rejected.
MULTI_FN_IR="./xls/dev_tools/testdata/benchmark_main_multi_fn.ir"
OPT_PROTO="${TEST_TMPDIR}/opt_options.txtpb"
CODEGEN_MATCH_PROTO="${TEST_TMPDIR}/codegen_match.txtpb"
CODEGEN_CONFLICT_PROTO="${TEST_TMPDIR}/codegen_conflict.txtpb"
echo -e 'top: "__benchmark_main_flags_test__inv"\narea_model: "asap7"' > "${OPT_PROTO}"
echo -e 'top: "__benchmark_main_flags_test__inv"\ngenerator: GENERATOR_KIND_COMBINATIONAL' > "${CODEGEN_MATCH_PROTO}"
echo -e 'top: "__benchmark_main_flags_test__add_with_inv"\ngenerator: GENERATOR_KIND_COMBINATIONAL' > "${CODEGEN_CONFLICT_PROTO}"

OPT_TOP_OUTPUT=$("${BENCHMARK_MAIN}" --opt_options_proto="${OPT_PROTO}" --run_evaluators=false "${MULTI_FN_IR}") || die "Failed running benchmark_main with --opt_options_proto"
echo "${OPT_TOP_OUTPUT}" | grep -F "Total area: 0.0437 um2" || die "Expected inv Total area 0.0437 um2 when --opt_options_proto sets top=__benchmark_main_flags_test__inv, got: ${OPT_TOP_OUTPUT}"

MATCH_TOP_OUTPUT=$("${BENCHMARK_MAIN}" --opt_options_proto="${OPT_PROTO}" --codegen_options_proto="${CODEGEN_MATCH_PROTO}" --run_evaluators=false "${MULTI_FN_IR}") || die "Failed running benchmark_main with matching top protos"
echo "${MATCH_TOP_OUTPUT}" | grep -F "Total area: 0.0437 um2" || die "Expected inv Total area 0.0437 um2 with matching top protos, got: ${MATCH_TOP_OUTPUT}"

if CONFLICT_OUTPUT=$("${BENCHMARK_MAIN}" --opt_options_proto="${OPT_PROTO}" --codegen_options_proto="${CODEGEN_CONFLICT_PROTO}" --run_evaluators=false "${MULTI_FN_IR}" 2>&1); then
  die "Expected benchmark_main to reject conflicting opt/codegen top protos, got: ${CONFLICT_OUTPUT}"
fi
echo "${CONFLICT_OUTPUT}" | grep -F "Conflicting top entities" || die "Expected conflicting top error message, got: ${CONFLICT_OUTPUT}"

echo "PASS"
