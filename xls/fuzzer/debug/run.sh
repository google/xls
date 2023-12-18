#!/bin/sh -e
# Copyright 2023 The XLS Authors
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

# Script for helping isolate LLVM JIT failures and suspected LLVM bugs. Takes
# XLS IR input and uses the JIT to generate LLVM IR which is invoked in several
# ways (with/without optimizations).
#
# This script is primarially illustrative and should be modified as appropriate
# for each persons debugging task.
#
# Usage:
#  (1) Set SRC_TOP to top of source tree.
#  (2) Point XLS_IR to the XLS IR sample and set XLS_INPUT to the
#      value which triggers the issue.
#  (3) Tweak the script as appropriate to use your llvm tools.
#  (4) Run script. Artifacts will be dumped in a newly created temp directory.

SRC_TOP=$PWD
if [[ -z ${SRC_TOP} ]]; then
  echo "SRC_TOP must be set to the top directory of the source repo."
  exit 1
fi
BIN=${SRC_TOP}/bazel-bin

XLS_IR=xls/fuzzer/debug/test.ir
XLS_INPUT="bits[8]:0xef;bits[16]:0x1234"
LLVM_OPT_LEVEL=2

LLVM_BIN_DIR=${SRC_TOP}/bazel-bin/llvm/llvm-project/llvm
if [[ -z ${LLVM_BIN_DIR} ]]; then
  echo "LLVM_BIN_DIR must be set."
  echo "When built from source this is the bin subdirectory in the LLVM "
  echo "build directory. Internal to Google, this the directory "
  echo "containing the artifacts from "
  echo "building llvm/llvm-project/llvm:all."
  exit 1
fi

CLANG=/usr/bin/clang

echo "=== Building tools..."
cd ${SRC_TOP}
# TODO(allight): 2023-12-12: Really we should build and use clang here too.
# Getting this to work is a bit annoying however since by default it doesn't
# seem to have the library search paths configured?
bazel build -c opt xls/tools:all llvm/llvm-project/llvm:all
cd -

OUTDIR=`mktemp -d xls_fuzzer_debug.XXXXXX --tmpdir`
echo "=== Output directory: ${OUTDIR}"

echo "=== Evaluating XLS IR and dumping LLVM IR..."
MAIN_IR=${OUTDIR}/test.main.ll
DUMPED_LLVM_IR=${OUTDIR}/test.ll
DUMPED_LLVM_OPT_IR=${OUTDIR}/jitopt.test.ll
${BIN}/xls/tools/eval_ir_main \
  --input="${XLS_INPUT}" \
  --use_llvm_jit \
  --llvm_jit_ir_output=${DUMPED_LLVM_IR} \
  --llvm_jit_opt_ir_output=${DUMPED_LLVM_OPT_IR} \
  --llvm_opt_level=${LLVM_OPT_LEVEL} \
  --llvm_jit_main_wrapper_output=${MAIN_IR} \
  --llvm_jit_main_wrapper_write_is_linked \
  ${XLS_IR}

echo "=== Optimizing LLVM IR"
OPT_LLVM_IR=${OUTDIR}/opt.test.ll
INLINE_LLVM_IR=${OUTDIR}/inlined.test.ll
OPT_INLINE_LLVM_IR=${OUTDIR}/opt.inlined.test.ll
${LLVM_BIN_DIR}/opt ${DUMPED_LLVM_IR} -O${LLVM_OPT_LEVEL} -S > ${OPT_LLVM_IR}
${LLVM_BIN_DIR}/opt ${DUMPED_LLVM_IR} -passes=inline,dce -S > ${INLINE_LLVM_IR}
${LLVM_BIN_DIR}/opt ${INLINE_LLVM_IR} -O${LLVM_OPT_LEVEL} -S > ${OPT_INLINE_LLVM_IR}

function build_and_run() {
  echo "=== $2 [$1]: Building assembly and object file from LLVM IR..."
  LLVM_IR=${OUTDIR}/$1.ll
  TEST_S=${OUTDIR}/$1.s
  TEST_O=${OUTDIR}/$1.o

  ${LLVM_BIN_DIR}/llc ${LLVM_IR} -O${LLVM_OPT_LEVEL} -filetype=asm -o ${TEST_S}
  ${LLVM_BIN_DIR}/llc ${LLVM_IR} -O${LLVM_OPT_LEVEL} -filetype=obj -o ${TEST_O}

  echo "=== $2 [$1]: Building binary..."
  ${CLANG} ${TEST_O} ${MAIN_IR} -o ${OUTDIR}/$1.main

  echo "=== $2 [$1]: Running binary..."
  ${OUTDIR}/$1.main | hd
}

build_and_run "test" "Unoptimized"
build_and_run "opt.test" "Optimized at -O${LLVM_OPT_LEVEL}"
build_and_run "inlined.test" "Only inlined"
build_and_run "opt.inlined.test" "Inlined then optimized at -O${LLVM_OPT_LEVEL}"

function link_and_run_ll() {
  LLVM_IR=${OUTDIR}/$1.ll
  COMBINED_IR=${OUTDIR}/$1.combined.ll

  echo "=== $2 [$1]: Linking standalone LLVM IR..."
  ${LLVM_BIN_DIR}/llvm-link ${LLVM_IR} ${MAIN_IR} -S -o ${COMBINED_IR}

  echo "=== $2 [$1]: Running with lli..."
  ${LLVM_BIN_DIR}/lli ${LLVM_OPTS} ${COMBINED_IR} | hd
}

link_and_run_ll "test" "Unoptimized"
link_and_run_ll "opt.test" "Optimized at -O${LLVM_OPT_LEVEL}"
link_and_run_ll "inlined.test" "Only inlined"
link_and_run_ll "opt.inlined.test" "Inlined then optimized at -O${LLVM_OPT_LEVEL}"
