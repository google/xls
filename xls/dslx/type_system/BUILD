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

# Frontend type-system and associated data structures.

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
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        ":parametric_expression",
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits_ops",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":parametric_expression",
        ":type",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "deduce_ctx",
    srcs = ["deduce_ctx.cc"],
    hdrs = ["deduce_ctx.h"],
    deps = [
        ":ast_env",
        ":parametric_env",
        ":parametric_expression",
        ":type",
        ":type_and_parametric_env",
        ":type_info",
        ":type_mismatch_error_data",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:import_routines",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "deduce",
    srcs = [
        "deduce.cc",
        "deduce_colon_ref.cc",
        "deduce_colon_ref.h",
        "deduce_enum_def.cc",
        "deduce_enum_def.h",
        "deduce_expr.cc",
        "deduce_expr.h",
        "deduce_invocation.cc",
        "deduce_invocation.h",
        "deduce_spawn.cc",
        "deduce_spawn.h",
        "deduce_struct_def_base_utils.cc",
        "deduce_struct_def_base_utils.h",
        "deduce_struct_instance.cc",
        "deduce_struct_instance.h",
    ],
    hdrs = ["deduce.h"],
    deps = [
        ":ast_env",
        ":deduce_ctx",
        ":deduce_utils",
        ":parametric_env",
        ":parametric_expression",
        ":parametric_instantiator",
        ":parametric_with_type",
        ":scoped_fn_stack_entry",
        ":type",
        ":type_and_parametric_env",
        ":type_info",
        ":type_zero_value",
        ":unwrap_meta_type",
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/exhaustiveness:match_exhaustiveness_checker",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:token_utils",
        "//xls/ir:bits",
        "//xls/ir:format_strings",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "parametric_with_type",
    srcs = ["parametric_with_type.cc"],
    hdrs = ["parametric_with_type.h"],
    deps = [
        ":type",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "parametric_instantiator_internal",
    srcs = ["parametric_instantiator_internal.cc"],
    hdrs = ["parametric_instantiator_internal.h"],
    deps = [
        ":deduce_ctx",
        ":parametric_bind",
        ":parametric_env",
        ":parametric_expression",
        ":parametric_with_type",
        ":scoped_fn_stack_entry",
        ":type",
        ":type_and_parametric_env",
        ":type_info",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/bytecode:bytecode_interpreter_options",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "parametric_instantiator",
    srcs = ["parametric_instantiator.cc"],
    hdrs = ["parametric_instantiator.h"],
    deps = [
        ":deduce_ctx",
        ":parametric_instantiator_internal",
        ":parametric_with_type",
        ":type",
        ":type_and_parametric_env",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "type_mismatch_error_data",
    hdrs = ["type_mismatch_error_data.h"],
    deps = [
        ":type",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "instantiate_parametric_function",
    srcs = ["instantiate_parametric_function.cc"],
    hdrs = ["instantiate_parametric_function.h"],
    deps = [
        ":deduce_ctx",
        ":deduce_utils",
        ":parametric_instantiator",
        ":parametric_with_type",
        ":type",
        ":type_and_parametric_env",
        ":type_info",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "typecheck_invocation",
    srcs = ["typecheck_invocation.cc"],
    hdrs = ["typecheck_invocation.h"],
    deps = [
        ":ast_env",
        ":deduce_ctx",
        ":deduce_utils",
        ":instantiate_parametric_function",
        ":parametric_env",
        ":parametric_with_type",
        ":scoped_fn_stack_entry",
        ":type",
        ":type_and_parametric_env",
        ":type_info",
        ":unwrap_meta_type",
        "//xls/common:casts",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:dslx_builtins_signatures",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@re2",
    ],
)

cc_library(
    name = "typecheck_module",
    srcs = ["typecheck_module.cc"],
    hdrs = ["typecheck_module.h"],
    deps = [
        ":deduce",
        ":deduce_ctx",
        ":scoped_fn_stack_entry",
        ":type",
        ":type_info",
        ":type_mismatch_error_data",
        ":typecheck_function",
        ":typecheck_invocation",
        ":unwrap_meta_type",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx/diagnostics:maybe_explain_error",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:proc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "typecheck_test_utils",
    testonly = True,
    srcs = ["typecheck_test_utils.cc"],
    hdrs = ["typecheck_test_utils.h"],
    deps = [
        ":type_info_to_proto",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
        "@re2",
    ],
)

cc_test(
    name = "typecheck_module_test",
    srcs = ["typecheck_module_test.cc"],
    deps = [
        ":type_info",
        ":typecheck_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:error_test_utils",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "proc_typecheck_test",
    srcs = ["proc_typecheck_test.cc"],
    deps = [
        ":typecheck_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:error_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "impl_typecheck_test",
    srcs = ["impl_typecheck_test.cc"],
    deps = [
        ":typecheck_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "parametric_expression",
    srcs = ["parametric_expression.cc"],
    hdrs = ["parametric_expression.h"],
    deps = [
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.cc"],
    deps = [
        ":parametric_expression",
        "//xls/common:xls_gunit_main",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_set",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "type_zero_value",
    srcs = ["type_zero_value.cc"],
    hdrs = ["type_zero_value.h"],
    deps = [
        ":type",
        ":type_info",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "parametric_env",
    srcs = ["parametric_env.cc"],
    hdrs = ["parametric_env.h"],
    deps = [
        "//xls/dslx:interp_value",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ast_env",
    srcs = ["ast_env.cc"],
    hdrs = ["ast_env.h"],
    deps = [
        "//xls/common:visitor",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:proc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:variant",
    ],
)

proto_library(
    name = "type_info_proto",
    srcs = ["type_info.proto"],
)

cc_proto_library(
    name = "type_info_cc_proto",
    deps = [":type_info_proto"],
)

cc_library(
    name = "type_info_to_proto",
    srcs = ["type_info_to_proto.cc"],
    hdrs = ["type_info_to_proto.h"],
    deps = [
        ":parametric_expression",
        ":type",
        ":type_info",
        ":type_info_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "type_info_to_proto_test",
    srcs = ["type_info_to_proto_test.cc"],
    data = glob(["testdata/*.txt"]),
    deps = [
        ":type_info_cc_proto",
        ":type_info_to_proto",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "type_and_parametric_env",
    hdrs = ["type_and_parametric_env.h"],
    deps = [
        ":parametric_env",
        ":type",
    ],
)

cc_library(
    name = "type_info",
    srcs = ["type_info.cc"],
    hdrs = ["type_info.h"],
    deps = [
        ":parametric_env",
        ":type",
        "//xls/common:symbolized_stacktrace",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "type_info_test",
    srcs = ["type_info_test.cc"],
    deps = [
        ":parametric_env",
        ":type_info",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "parametric_bind",
    srcs = ["parametric_bind.cc"],
    hdrs = ["parametric_bind.h"],
    deps = [
        ":deduce_ctx",
        ":parametric_expression",
        ":type",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "parametric_bind_test",
    srcs = ["parametric_bind_test.cc"],
    deps = [
        ":deduce_ctx",
        ":parametric_bind",
        ":parametric_expression",
        ":type",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "unwrap_meta_type",
    srcs = ["unwrap_meta_type.cc"],
    hdrs = ["unwrap_meta_type.h"],
    deps = [
        ":type",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# -- binary for invoking type checking subsystem

cc_binary(
    name = "typecheck_main",
    srcs = ["typecheck_main.cc"],
    deps = [
        ":type_info_cc_proto",
        ":type_info_to_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:error_printer",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:bindings",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

py_test(
    name = "typecheck_main_test",
    srcs = ["typecheck_main_test.py"],
    data = [
        ":typecheck_main",
        "//xls/dslx/tests:mod_const_enum_importer.x",
        "//xls/dslx/tests:mod_simple_const_enum.x",
    ],
    deps = [
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
    ],
)

cc_library(
    name = "scoped_fn_stack_entry",
    srcs = ["scoped_fn_stack_entry.cc"],
    hdrs = ["scoped_fn_stack_entry.h"],
    deps = [
        ":deduce_ctx",
        ":parametric_env",
        "//xls/dslx/frontend:ast",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "deduce_utils",
    srcs = ["deduce_utils.cc"],
    hdrs = ["deduce_utils.h"],
    deps = [
        ":deduce_ctx",
        ":parametric_with_type",
        ":type",
        ":type_info",
        ":unwrap_meta_type",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:token_utils",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
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
    name = "deduce_utils_test",
    srcs = ["deduce_utils_test.cc"],
    deps = [
        ":deduce_utils",
        ":type",
        ":type_info",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/frontend:scanner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "typecheck_function",
    srcs = ["typecheck_function.cc"],
    hdrs = ["typecheck_function.h"],
    deps = [
        ":deduce",
        ":deduce_ctx",
        ":parametric_env",
        ":scoped_fn_stack_entry",
        ":type",
        ":type_info",
        ":typecheck_invocation",
        ":unwrap_meta_type",
        "//xls/common:casts",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/diagnostics:warn_on_defined_but_unused",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "zip_types",
    srcs = ["zip_types.cc"],
    hdrs = ["zip_types.h"],
    deps = [
        ":type",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "zip_types_test",
    srcs = ["zip_types_test.cc"],
    deps = [
        ":type",
        ":zip_types",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)
