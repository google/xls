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

# Verify that block evaluation elaborates the top container block rather than
# blocks()[0] when benchmarking a multi-proc package.
MULTI_PROC_CYCLE_IR="${TEST_TMPDIR}/multi_proc_cycle.ir"
cat > "${MULTI_PROC_CYCLE_IR}" <<EOF
package multi_proc_cycle

chan a_to_b(bits[8], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive, fifo_depth=0, bypass=true, register_push_outputs=false, register_pop_outputs=false)
chan b_to_a(bits[8], id=1, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive, fifo_depth=0, bypass=true, register_push_outputs=false, register_pop_outputs=false)

proc child_b(state: (), init={()}) {
  t0: token = literal(value=token, id=1)
  rcv: (token, bits[8]) = receive(t0, channel=a_to_b, id=2)
  rcv_tok: token = tuple_index(rcv, index=0, id=3)
  rcv_data: bits[8] = tuple_index(rcv, index=1, id=4)
  snd: token = send(rcv_tok, rcv_data, channel=b_to_a, id=5)
  next (state)
}

top proc top_a(state: (), init={()}) {
  t0: token = literal(value=token, id=10)
  rcv: (token, bits[8]) = receive(t0, channel=b_to_a, id=11)
  rcv_tok: token = tuple_index(rcv, index=0, id=12)
  rcv_data: bits[8] = tuple_index(rcv, index=1, id=13)
  snd: token = send(rcv_tok, rcv_data, channel=a_to_b, id=14)
  next (state)
}
EOF

if "${BENCHMARK_MAIN}" --delay_model=unit --pipeline_stages=1 --reset=rst --multi_proc=true --run_evaluators=true "${MULTI_PROC_CYCLE_IR}" > /dev/null 2> "${TEST_TMPDIR}/cycle.err"; then
  die "Expected benchmark_main to fail on combinational cycle in top container block"
fi
grep -F "Cycle detected" "${TEST_TMPDIR}/cycle.err" || die "Expected Cycle detected error, got: $(cat "${TEST_TMPDIR}/cycle.err")"

echo "PASS"
