# Copyright 2023 The XLS Authors
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

load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_library")
load(
    "//xls/build_rules:xls_macros.bzl",
    "xls_dslx_cpp_type_library",
)

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "cpp_transpiler",
    srcs = ["cpp_transpiler.cc"],
    hdrs = ["cpp_transpiler.h"],
    deps = [
        ":cpp_type_generator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_library(
    name = "cpp_emitter",
    srcs = ["cpp_emitter.cc"],
    hdrs = ["cpp_emitter.h"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/common:case_converters",
        "//xls/common:indent",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_library(
    name = "cpp_type_generator",
    srcs = ["cpp_type_generator.cc"],
    hdrs = ["cpp_type_generator.h"],
    deps = [
        ":cpp_emitter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/common:indent",
        "//xls/common:visitor",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_test(
    name = "cpp_transpiler_test",
    srcs = ["cpp_transpiler_test.cc"],
    data = glob([
        "testdata/*",
    ]),
    deps = [
        ":cpp_transpiler",
        ":cpp_type_generator",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:golden_files",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "cpp_transpiler_main",
    srcs = ["cpp_transpiler_main.cc"],
    visibility = ["//xls:xls_public"],
    deps = [
        ":cpp_transpiler",
        ":cpp_type_generator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_kind",
    ],
)

xls_dslx_cpp_type_library(
    name = "test_types_lib",
    src = ":test_types.x",
    namespace = "xls::test",
    deps = [":test_types"],
)

cc_test(
    name = "test_types_test",
    srcs = ["test_types_test.cc"],
    deps = [
        ":test_types_lib",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

xls_dslx_library(
    name = "test_types_dep",
    srcs = ["test_types_dep.x"],
)

xls_dslx_library(
    name = "test_types",
    srcs = ["test_types.x"],
    deps = [":test_types_dep"],
)
