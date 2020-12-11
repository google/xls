#!/bin/bash
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
set -x

XLS_HOME_ABS=$TEST_SRCDIR/com_google_xls/xls
XLS_HOME_REL=xls
OUTPUT_PATH=$TEST_TMPDIR/generated.x

${XLS_HOME_REL}/tools/proto_to_dslx_main \
  -proto_def_path ${XLS_HOME_REL}/tools/testdata/proto_to_dslx_main.proto     \
  -proto_name xls.Fields                                                      \
  -textproto_path ${XLS_HOME_REL}/tools/testdata/proto_to_dslx_main.textproto \
  -var_name Foo                                                               \
  -output_path $OUTPUT_PATH || exit -1

# Verify the output matches the expected.
diff $XLS_HOME_ABS/tools/testdata/proto_to_dslx_main.x \
     $OUTPUT_PATH || exit -1

# Verify it parses..
$XLS_HOME_ABS/dslx/parser_main $OUTPUT_PATH || exit -1

# And verify it interprets.
$XLS_HOME_ABS/dslx/interpreter/interpreter_main $OUTPUT_PATH || exit -1

echo "PASS!"
