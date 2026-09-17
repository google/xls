# Copyright 2026 The XLS Authors
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

# Build rules for XLS spin examples.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

exports_files(glob(include = ["*.x"]))

# counter

xls_dslx_library(
    name = "counter_dslx",
    srcs = ["counter.x"],
)

xls_dslx_test(
    name = "counter_dslx_guided_test",
    library = ":counter_dslx",
    dslx_test_args = {
        "guided_model_check": "True"
    }
)

xls_dslx_test(
    name = "counter_dslx_exhaustive_test",
    library = ":counter_dslx",
    dslx_test_args = {
        "exhaustive_model_check": "True"
    }
)

# order_dependence

xls_dslx_library(
    name = "order_dependence_dslx",
    srcs = ["order_dependence.x"],
)

xls_dslx_test(
    name = "order_dependence_guided_test",
    library = ":order_dependence_dslx",
    dslx_test_args = {
        "guided_model_check": "True"
    },
    tags = ["manual"],  # expected to fail: SPIN guided path finds the race DSLX misses
)

xls_dslx_test(
    name = "order_dependence_exhaustive_test",
    library = ":order_dependence_dslx",
    dslx_test_args = {
        "exhaustive_model_check": "True"
    },
    tags = ["manual"],  # expected to fail: demonstrates SPIN finding the race
)
