# Copyright 2021 The XLS Authors
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

# Integration tests for the generated JIT wrapper being invoked from C++ code.

load("//xls/build_rules:build_defs.bzl", "dslx_jit_wrapper", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

dslx_test(
    name = "simple",
    srcs = ["simple.x"],
    # Disable these because there's no main entry point to speak of.
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_jit_wrapper(
    name = "identity",
    dslx_name = "F32Identity",
    entry_function = "identity",
    deps = [":simple_opt_ir"],
)

dslx_jit_wrapper(
    name = "is_inf",
    dslx_name = "F32IsInf",
    entry_function = "is_inf",
    deps = [":simple_opt_ir"],
)

dslx_jit_wrapper(
    name = "fail_on_42",
    dslx_name = "FailOn42",
    entry_function = "fail_on_42",
    deps = [":simple_opt_ir"],
)

cc_test(
    name = "simple_jit_wrapper_test",
    srcs = ["simple_jit_wrapper_test.cc"],
    deps = [
        ":fail_on_42",
        ":identity",
        ":is_inf",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value_helpers",
        "@com_google_googletest//:gtest",
    ],
)
