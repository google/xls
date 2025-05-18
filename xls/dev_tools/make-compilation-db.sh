#!/usr/bin/env bash
# Copyright 2023 The XLS Authors.
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

set -u
set -e

# Which bazel and bant to use. If unset, defaults are used.
BAZEL=${BAZEL:-bazel}
BANT=$($(dirname $0)/get-bant-path.sh)

# Important to run with --remote_download_outputs=all to make sure generated
# files are actually visible locally in case a remote cache (that includes
# --disk_cache) is used ( https://github.com/hzeller/bazel-gen-file-issue )
BAZEL_OPTS="-c opt --remote_download_outputs=all"

# Tickle some build targets to fetch all dependencies and generate files,
# so that they can be seen by the users of the compilation db.
#
# Use bant to find some genrule-like targets (such as cc_proto_library)
# and execute all these, and choose a bunch of other targets specifically
# to provide a complete set of all generated files.
# If more targets are needed here, try to use the smallest set that covers
# the needed files to avoid triggering an unnecessarily large build.
"${BAZEL}" build ${BAZEL_OPTS} \
  @linenoise @nlohmann_json//:singleheader-json \
  @zstd @at_clifford_yosys//:json11 \
  @verible//verible/common/lsp:lsp-protocol.h \
  //xls/common:xls_gunit_main \
  //xls/solvers:z3_ir_translator \
  //xls/dslx/tests/trace_fmt_issue_651:trace_{u16,u21,s32,enum,u16_hex,u21_hex}_wrapper \
  $("${BANT}" list-targets @com_google_ortools//... | awk '/cc_proto_library/ {print $3}') \
  $("${BANT}" list-targets | \
    awk '/cc_proto_library|xls_dslx_cpp_type_library|cc_xls_ir_jit_wrapper|xls_ir_cc_library|gentbl_cc_library|cc_grpc_library/ {print $3}')

# Create compilation DB. Command 'compilation-db' creates a huge *.json file,
# but compile_flags.txt is perfectly sufficient and easier for tools to use.
"${BANT}" compile-flags > compile_flags.txt

# If there are two styles of comp-dbs, tools might have issues. Warn user.
if [ -r compile_commands.json ]; then
  echo -e "\n\033[1;31mSuggest to remove old compile_commands.json to not interfere with compile_flags.txt\033[0m\n"
fi
