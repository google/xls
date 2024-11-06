# Copyright 2022 The XLS Authors
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
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ir_wrapper",
    srcs = ["ir_wrapper.cc"],
    hdrs = ["ir_wrapper.h"],
    deps = [
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:mangle",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/ir_convert:conversion_info",
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/jit:function_jit",
        "//xls/jit:jit_channel_queue",
        "//xls/jit:jit_proc_runtime",
        "//xls/jit:jit_runtime",
        "//xls/passes:optimization_pass_pipeline",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ir_wrapper_test",
    srcs = ["ir_wrapper_test.cc"],
    data = glob(["testdata/*.ir"]),
    deps = [
        ":ir_wrapper",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:module",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_view",
        "//xls/jit:function_jit",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)
