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

BINPATH=./xls/dev_tools/parse_ir
$BINPATH <<EOF
wheeeeeeeee
EOF

RETCODE=$?
if [[ ${RETCODE} -ne 0 ]]; then
  echo "Bad contents gave retcode ${RETCODE}.";
else
  echo "Bad contents gave ok retcode! :-(";
  exit -1
fi

$BINPATH <<EOF || exit -1
package simple

fn id(x: bits[2]) -> bits[2] {
  ret param.2: bits[2] = param(name=x)
}
EOF

$BINPATH ./xls/dev_tools/testdata/add_folding_overlarge.ir || exir -1

$BINPATH ./xls/dev_tools/testdata/file_that_definitely_does_not_or_at_least_should_not_exist.ir
RETCODE=$?
if [[ ${RETCODE} -ne 0 ]]; then
  echo "Bad path gave retcode ${RETCODE}.";
else
  echo "Bad path gave ok retcode! :-(";
  exit -1
fi

echo "PASS!";
