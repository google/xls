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

# Conversion from frontend representation to XLS IR.

# pytype tests are present in this file
# Load proto_library
# cc_proto_library is used in this file

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
    name = "channel_scope",
    srcs = ["channel_scope.cc"],
    hdrs = ["channel_scope.h"],
    deps = [
        ":conversion_info",
        ":ir_conversion_utils",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:name_uniquer",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "channel_scope_test",
    srcs = ["channel_scope_test.cc"],
    deps = [
        ":channel_scope",
        ":conversion_info",
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:channel_direction",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/frontend:proc_test_utils",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir",
        "//xls/ir:channel_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "conversion_info",
    hdrs = ["conversion_info.h"],
    deps = [
        "//xls/ir",
        "//xls/ir:xls_ir_interface_cc_proto",
    ],
)

cc_library(
    name = "ir_conversion_utils",
    srcs = ["ir_conversion_utils.cc"],
    hdrs = ["ir_conversion_utils.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/type_system:deduce_ctx",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:parametric_expression",
        "//xls/dslx/type_system:type",
        "//xls/ir",
        "//xls/ir:type",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "ir_conversion_utils_test",
    srcs = ["ir_conversion_utils_test.cc"],
    deps = [
        ":ir_conversion_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:parametric_expression",
        "//xls/dslx/type_system:type",
        "//xls/ir",
        "//xls/ir:type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "proc_config_ir_converter",
    srcs = ["proc_config_ir_converter.cc"],
    hdrs = ["proc_config_ir_converter.h"],
    deps = [
        ":channel_scope",
        ":conversion_info",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "proc_config_ir_converter_test",
    srcs = ["proc_config_ir_converter_test.cc"],
    deps = [
        ":channel_scope",
        ":conversion_info",
        ":convert_options",
        ":ir_converter",
        ":proc_config_ir_converter",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/type_system:parametric_env",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:ir_matcher",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "ir_converter_test",
    srcs = ["ir_converter_test.cc"],
    data = glob(["testdata/*.ir"]),
    deps = [
        ":convert_options",
        ":ir_converter",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/run_routines",
        "//xls/dslx/run_routines:run_comparator",
        "//xls/dslx/type_system:typecheck_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
        "@re2",
    ],
)

filegroup(
    name = "ir_converter_test_sh",
    srcs = ["ir_converter_test.sh"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "convert_options",
    hdrs = ["convert_options.h"],
    deps = [
        "//xls/dslx:warning_kind",
        "//xls/ir:channel",
    ],
)

cc_library(
    name = "function_converter",
    srcs = ["function_converter.cc"],
    hdrs = ["function_converter.h"],
    data = glob(["testdata/*.ir"]),
    deps = [
        ":channel_scope",
        ":conversion_info",
        ":convert_format_macro",
        ":convert_options",
        ":ir_conversion_utils",
        ":proc_config_ir_converter",
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:mangle",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:foreign_function",
        "//xls/ir:function_builder",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:verifier",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "function_converter_test",
    srcs = ["function_converter_test.cc"],
    deps = [
        ":conversion_info",
        ":convert_options",
        ":function_converter",
        "//xls/common:golden_files",
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/ir",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ir_converter",
    srcs = ["ir_converter.cc"],
    hdrs = ["ir_converter.h"],
    deps = [
        ":channel_scope",
        ":conversion_info",
        ":convert_options",
        ":extract_conversion_order",
        ":function_converter",
        ":proc_config_ir_converter",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:typecheck_module",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_scanner",
        "//xls/ir:value",
        "//xls/ir:verifier",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@cppitertools",
    ],
)

cc_library(
    name = "extract_conversion_order",
    srcs = ["extract_conversion_order.cc"],
    hdrs = ["extract_conversion_order.h"],
    deps = [
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/frontend:proc_id",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "extract_conversion_order_test",
    srcs = ["extract_conversion_order_test.cc"],
    deps = [
        ":extract_conversion_order",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/type_system:parametric_env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "convert_format_macro",
    srcs = ["convert_format_macro.cc"],
    hdrs = ["convert_format_macro.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:make_value_format_descriptor",
        "//xls/dslx:value_format_descriptor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:bits",
        "//xls/ir:format_preference",
        "//xls/ir:format_strings",
        "//xls/ir:function_builder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

proto_library(
    name = "ir_converter_options_flags_proto",
    srcs = ["ir_converter_options_flags.proto"],
    deps = ["//xls/ir:channel_proto"],
)

cc_proto_library(
    name = "ir_converter_options_flags_cc_proto",
    deps = [":ir_converter_options_flags_proto"],
)

cc_library(
    name = "ir_converter_options_flags",
    srcs = ["ir_converter_options_flags.cc"],
    hdrs = ["ir_converter_options_flags.h"],
    deps = [
        ":ir_converter_options_flags_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/ir:channel_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "ir_converter_options_flags_test",
    srcs = ["ir_converter_options_flags_test.cc"],
    deps = [
        ":ir_converter_options_flags",
        ":ir_converter_options_flags_cc_proto",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@googletest//:gtest",
    ],
)

cc_binary(
    name = "ir_converter_main",
    srcs = ["ir_converter_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":conversion_info",
        ":convert_options",
        ":ir_converter",
        ":ir_converter_options_flags",
        ":ir_converter_options_flags_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:warning_kind",
        "//xls/ir",
        "//xls/ir:channel",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

filegroup(
    name = "sv_types_testfiles",
    testonly = True,
    srcs = glob(["testdata/sv_types/*.x"]),
    visibility = ["//visibility:private"],
)

py_test(
    name = "ir_converter_main_test",
    srcs = ["ir_converter_main_test.py"],
    data = [
        ":ir_converter_main",
        ":sv_types_testfiles",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/ir:xls_ir_interface_py_pb2",
        "//xls/ir:xls_type_py_pb2",
    ],
)
