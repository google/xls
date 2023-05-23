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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ir_conversion_utils",
    srcs = ["ir_conversion_utils.cc"],
    hdrs = ["ir_conversion_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:deduce_ctx",
        "//xls/dslx/type_system:parametric_env",
        "//xls/ir",
        "//xls/ir:type",
    ],
)

cc_test(
    name = "ir_conversion_utils_test",
    srcs = ["ir_conversion_utils_test.cc"],
    deps = [
        ":ir_conversion_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:parametric_expression",
        "//xls/ir",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_config_ir_converter",
    srcs = ["proc_config_ir_converter.cc"],
    hdrs = ["proc_config_ir_converter.h"],
    deps = [
        ":extract_conversion_order",
        ":ir_conversion_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "//xls/ir",
        "//xls/ir:channel",
    ],
)

cc_test(
    name = "proc_config_ir_converter_test",
    srcs = ["proc_config_ir_converter_test.cc"],
    deps = [
        ":proc_config_ir_converter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/ir",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ir_converter_test",
    srcs = ["ir_converter_test.cc"],
    data = glob(["testdata/*.ir"]),
    deps = [
        ":ir_converter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:golden_files",
        "//xls/common:init_xls",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_googletest//:gtest",
    ],
)

filegroup(
    name = "ir_converter_test_sh",
    srcs = ["ir_converter_test.sh"],
    visibility = ["//xls:xls_users"],
)

cc_library(
    name = "convert_options",
    hdrs = ["convert_options.h"],
)

cc_library(
    name = "function_converter",
    srcs = ["function_converter.cc"],
    hdrs = ["function_converter.h"],
    deps = [
        ":convert_format_macro",
        ":convert_options",
        ":ir_conversion_utils",
        ":proc_config_ir_converter",
        "@com_google_absl//absl/status",
        "//xls/common:visitor",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:import_data",
        "//xls/dslx:mangle",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/ir",
        "//xls/ir:function_builder",
    ],
)

cc_test(
    name = "function_converter_test",
    srcs = ["function_converter_test.cc"],
    deps = [
        ":function_converter",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/ir",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ir_converter",
    srcs = ["ir_converter.cc"],
    hdrs = ["ir_converter.h"],
    deps = [
        ":convert_format_macro",
        ":convert_options",
        ":extract_conversion_order",
        ":function_converter",
        ":ir_conversion_utils",
        ":proc_config_ir_converter",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:visitor",
        "//xls/common/file:filesystem",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:make_value_format_descriptor",
        "//xls/dslx:mangle",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:typecheck",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "extract_conversion_order",
    srcs = ["extract_conversion_order.cc"],
    hdrs = ["extract_conversion_order.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_test(
    name = "extract_conversion_order_test",
    srcs = ["extract_conversion_order_test.cc"],
    deps = [
        ":extract_conversion_order",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "convert_format_macro",
    srcs = ["convert_format_macro.cc"],
    hdrs = ["convert_format_macro.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "//xls/dslx:make_value_format_descriptor",
        "//xls/dslx:value_format_descriptor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:function_builder",
    ],
)

cc_binary(
    name = "ir_converter_main",
    srcs = ["ir_converter_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":ir_converter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

py_test(
    name = "ir_converter_main_test",
    srcs = ["ir_converter_main_test.py"],
    data = [":ir_converter_main"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)
