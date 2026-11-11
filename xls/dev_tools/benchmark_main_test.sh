#!/usr/bin/env bash
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MY_BENCHMARK_FILE="./xls/examples/proc_network_opt_ir_benchmark.sh"
BENCHMARK_MAIN="./xls/dev_tools/benchmark_main"
ABORT_PROC_IR="./xls/dev_tools/testdata/benchmark_main_fail_on_tick.ir"
DEADLOCK_PROC_IR="./xls/dev_tools/testdata/benchmark_main_deadlock.ir"
FRACTIONAL_AREA_IR="./xls/dev_tools/testdata/benchmark_main_inv.ir"

die () {
  echo "ERROR: $1"
  exit 1
}

# Run the proc network benchmark and verify non-zero Kcalls/s & calls/s rates.
PROC_OUTPUT=$("${MY_BENCHMARK_FILE}") || die "Failed in ${MY_BENCHMARK_FILE}"
echo "${PROC_OUTPUT}" | grep -E "JIT run time \(unoptimized\): [1-9][0-9]* Kcalls/s" || die "Expected non-zero Proc JIT Kcalls/s (unoptimized), got: ${PROC_OUTPUT}"
echo "${PROC_OUTPUT}" | grep -E "Interpreter run time \(unoptimized\): [1-9][0-9]* calls/s" || die "Expected non-zero Proc Interpreter calls/s (unoptimized), got: ${PROC_OUTPUT}"
echo "${PROC_OUTPUT}" | grep -E "JIT run time \(optimized\): [1-9][0-9]* Kcalls/s" || die "Expected non-zero Proc JIT Kcalls/s (optimized), got: ${PROC_OUTPUT}"
echo "${PROC_OUTPUT}" | grep -E "Interpreter run time \(optimized\): [1-9][0-9]* calls/s" || die "Expected non-zero Proc Interpreter calls/s (optimized), got: ${PROC_OUTPUT}"

# Verify procs that trigger fail!/assert! on random inputs abort evaluation
# cleanly without counting failed ticks or spinning.
ABORT_OUTPUT=$("${BENCHMARK_MAIN}" --area_model=asap7 "${ABORT_PROC_IR}") || die "Failed running benchmark_main on ${ABORT_PROC_IR}"
if echo "${ABORT_OUTPUT}" | grep -q "JIT run time"; then
  die "Expected aborted proc to skip JIT run time metric, got: ${ABORT_OUTPUT}"
fi

# Verify procs that deadlock after an initial tick break immediately instead of
# spinning on non-reset internal queues.
DEADLOCK_OUTPUT=$("${BENCHMARK_MAIN}" --area_model=asap7 "${DEADLOCK_PROC_IR}") || die "Failed running benchmark_main on ${DEADLOCK_PROC_IR}"
if echo "${DEADLOCK_OUTPUT}" | grep -q "JIT run time"; then
  die "Expected deadlocked proc to skip JIT run time metric, got: ${DEADLOCK_OUTPUT}"
fi

# Verify fractional square-micron area is preserved without integer truncation.
AREA_OUTPUT=$("${BENCHMARK_MAIN}" --area_model=asap7 --run_evaluators=false "${FRACTIONAL_AREA_IR}") || die "Failed running benchmark_main on ${FRACTIONAL_AREA_IR}"
echo "${AREA_OUTPUT}" | grep -F "Total area: 0.0437 um2" || die "Expected fractional Total area 0.0437 um2, got: ${AREA_OUTPUT}"

echo "PASS"
