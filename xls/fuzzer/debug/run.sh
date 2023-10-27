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

XLS_IR=test.ir
XLS_INPUT="bits[8]:0xef;bits[16]:0x1234"
MAIN_C=main.c

SRC_TOP=
if [[ -z ${SRC_TOP} ]]; then
  echo "SRC_TOP must be set to the top directory of the source repo."
  exit 1
fi
BIN=${SRC_TOP}/bazel-bin

LLVM_BIN_DIR=
if [[ -z ${LLVM_BIN_DIR} ]]; then
  echo "LLVM_BIN_DIR must be set."
  echo "When built from source this is the bin subdirectory in the LLVM "
  echo "build directory. Internal to Google, this the directory "
  echo "containing the artifacts from building llvm/llvm-project/llvm:all."
  exit 1
fi

LLVM_OPTS=-O0

CLANG=/usr/bin/clang

echo "=== Building XLS tools..."
cd ${SRC_TOP}
bazel build -c opt xls/tools:all
cd -

OUTDIR=`mktemp -d`
echo "=== Output directory: ${OUTDIR}"

echo "=== Evaluating XLS IR and dumping LLVM IR..."
LLVM_IR=${OUTDIR}/test.ll
LLVM_OPT_IR=${OUTDIR}/test.opt.ll
${BIN}/xls/tools/eval_ir_main \
  --input=${XLS_INPUT} \
  --use_llvm_jit \
  --llvm_jit_ir_output=${LLVM_IR} \
  --llvm_jit_opt_ir_output=${LLVM_OPT_IR} \
  ${XLS_IR}

echo "=== Building assembly and object file from LLVM IR..."
TEST_S=${OUTDIR}/test.s
TEST_O=${OUTDIR}/test.o
${LLVM_BIN_DIR}/llc ${LLVM_IR} ${LLVM_OPTS} -filetype=asm -o ${TEST_S}
${LLVM_BIN_DIR}/llc ${LLVM_IR} ${LLVM_OPTS} -filetype=obj -o ${TEST_O}

echo "== Building binary..."
${CLANG} ${TEST_O} ${MAIN_C} -o ${OUTDIR}/main

echo "== Running binary..."
${OUTDIR}/main

# Commands for generating a LLVM-IR-only implementation and evaluating with lli.
# echo "== Compiling main to LLVM IR..."
# MAIN_IR=${OUTDIR}/main.ll
# ${CLANG} -S -emit-llvm ${MAIN_C} -o ${MAIN_IR}
# # Swap in the datalayout used in the JIT
# sed -i 's/^target datalayout.*/target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"/' ${OUTDIR}/main.ll

# echo "== Generating combined LLVM IR..."
# COMBINED_IR=${OUTDIR}/combined.ll
# ${LLVM_BIN_DIR}/llvm-link ${LLVM_OPT_IR} ${MAIN_IR} -S -o ${COMBINED_IR}

# echo "== Running with lli..."
# ${LLVM_BIN_DIR}/lli ${LLVM_OPTS} ${COMBINED_IR}
