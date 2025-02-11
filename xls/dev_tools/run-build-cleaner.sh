#!/usr/bin/env bash
# Copyright 2025 The XLS Authors.
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

readonly BANT_EXIT_ON_DWYU_ISSUES=3

# Which bazel and bant to use can be chosen by environment variables
BAZEL=${BAZEL:-bazel}
BANT=$($(dirname $0)/get-bant-path.sh)

# Run depend-on-what-you-use build-cleaner.
# Print buildifier commands to fix if needed.
"${BANT}" dwyu "$@"

BANT_EXIT=$?
if [ ${BANT_EXIT} -eq ${BANT_EXIT_ON_DWYU_ISSUES} ]; then
  cat >&2 <<EOF

Build dependency issues found, the following one-liner will fix it. Amend PR.

source <(xls/dev_tools/run-build-cleaner.sh $@)
EOF
fi

exit $BANT_EXIT
