# Copyright 2025 The XLS Authors
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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "verilog_fuzz_domain",
    testonly = True,
    srcs = ["verilog_fuzz_domain.cc"],
    hdrs = ["verilog_fuzz_domain.h"],
    deps = [
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/fuzzer/ir_fuzzer:ir_fuzz_domain",
        "//xls/ir",
        "//xls/public:runtime_build_actions",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "codegen_fuzz_test",
    srcs = ["codegen_fuzz_test.cc"],
    deps = [
        ":verilog_fuzz_domain",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)
