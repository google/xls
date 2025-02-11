# Copyright 2024 The XLS Authors
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

load("@bazel_skylib//rules:diff_test.bzl", "diff_test")
load("//xls/build_rules:xls_dslx_rules.bzl", "xls_dslx_to_verilog_package")

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
    name = "dslx_to_verilog",
    srcs = ["dslx_to_verilog.cc"],
    hdrs = ["dslx_to_verilog.h"],
    deps = [
        "//xls/codegen/vast",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:bits",
        "//xls/ir:name_uniquer",
        "//xls/ir:source_location",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "dslx_to_verilog_test",
    srcs = ["dslx_to_verilog_test.cc"],
    data = glob(["testdata/*"]),
    deps = [
        ":dslx_to_verilog",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_binary(
    name = "dslx_to_verilog_main",
    srcs = ["dslx_to_verilog_main.cc"],
    deps = [
        ":dslx_to_verilog",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:visitor",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/ir_convert:ir_converter_options_flags",
        "//xls/dslx/ir_convert:ir_converter_options_flags_cc_proto",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

# Target to test dslx_to_verilog_main against golden.
xls_dslx_to_verilog_package(
    name = "memory_proc_pkg",
    library = "//xls/examples:memory_proc_dslx",
    lint_waivers = ["struct-union-name-style"],
    namespace = "memory_proc",
    package_file = "memory_proc_pkg.sv",
)

diff_test(
    name = "dslx_to_verilog_main_memory_proc_test",
    file1 = "testdata/memory_proc_pkg.sv",
    file2 = ":memory_proc_pkg.sv",
)
