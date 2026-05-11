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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "dslx_fuzz_test",
    "xls_dslx_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "fuzz_test_dslx",
    srcs = ["fuzz_tests.x"],
)

dslx_fuzz_test(
    name = "test_arbitrary_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_arbitrary",
)

dslx_fuzz_test(
    name = "test_arbitrary_opt_fuzz_test",
    library = ":fuzz_test_dslx",
    opt_ir = True,
    test_function = "test_arbitrary",
)

dslx_fuzz_test(
    name = "test_range_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_range",
)

dslx_fuzz_test(
    name = "test_element_of_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_element_of",
)
