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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "inference_table",
    srcs = ["inference_table.cc"],
    hdrs = ["inference_table.h"],
    deps = [
        ":type_annotation_utils",
        "//xls/common:visitor",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "inference_table_converter",
    hdrs = ["inference_table_converter.h"],
    deps = [
        ":inference_table",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "inference_table_converter_impl",
    srcs = ["inference_table_converter_impl.cc"],
    hdrs = ["inference_table_converter_impl.h"],
    deps = [
        ":evaluator",
        ":expand_variables",
        ":inference_table",
        ":inference_table_converter",
        ":parametric_struct_instantiator",
        ":solve_for_parametrics",
        ":type_annotation_resolver",
        ":type_annotation_utils",
        ":type_system_tracer",
        ":unify_type_annotations",
        ":validate_concrete_type",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtin_stubs_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:type_zero_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "inference_table_test",
    srcs = ["inference_table_test.cc"],
    deps = [
        ":inference_table",
        ":inference_table_converter",
        ":inference_table_converter_impl",
        ":type_annotation_utils",
        ":type_system_test_utils",
        ":type_system_tracer",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "solve_for_parametrics",
    srcs = ["solve_for_parametrics.cc"],
    hdrs = ["solve_for_parametrics.h"],
    deps = [
        ":type_annotation_utils",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:zip_ast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "solve_for_parametrics_test",
    srcs = ["solve_for_parametrics_test.cc"],
    deps = [
        ":solve_for_parametrics",
        ":type_annotation_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "//xls/ir:bits",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "populate_table",
    srcs = ["populate_table.cc"],
    hdrs = ["populate_table.h"],
    deps = [
        ":inference_table",
        ":inference_table_converter",
        ":inference_table_converter_impl",
        ":type_annotation_utils",
        ":type_system_tracer",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:import_routines",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtin_stubs_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:type_info",
        "//xls/ir:format_strings",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "typecheck_module_v2_test",
    srcs = ["typecheck_module_v2_test.cc"],
    shard_count = 4,
    deps = [
        ":matchers",
        ":type_system_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/type_system:typecheck_test_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "typecheck_module_v2_builtin_test",
    srcs = ["typecheck_module_v2_builtin_test.cc"],
    deps = [
        ":matchers",
        "//xls/common:xls_gunit_main",
        "//xls/dslx:create_import_data",
        "//xls/dslx/type_system:typecheck_test_utils",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "type_system_test_utils",
    testonly = True,
    srcs = ["type_system_test_utils.cc"],
    hdrs = ["type_system_test_utils.h"],
    deps = [
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "type_annotation_utils",
    srcs = ["type_annotation_utils.cc"],
    hdrs = ["type_annotation_utils.h"],
    deps = [
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits",
        "//xls/ir:number_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "validate_concrete_type",
    srcs = ["validate_concrete_type.cc"],
    hdrs = ["validate_concrete_type.h"],
    deps = [
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:unwrap_meta_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "expand_variables",
    srcs = ["expand_variables.cc"],
    hdrs = ["expand_variables.h"],
    deps = [
        ":inference_table",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node_visitor_with_default",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "unify_type_annotations",
    srcs = ["unify_type_annotations.cc"],
    hdrs = ["unify_type_annotations.h"],
    deps = [
        ":evaluator",
        ":inference_table",
        ":parametric_struct_instantiator",
        ":type_annotation_utils",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "evaluator",
    hdrs = ["evaluator.h"],
    deps = [
        ":inference_table",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "type_system_tracer",
    srcs = ["type_system_tracer.cc"],
    hdrs = ["type_system_tracer.h"],
    deps = [
        ":inference_table",
        "//xls/common:indent",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "type_system_tracer_test",
    srcs = ["type_system_tracer_test.cc"],
    deps = [
        ":type_annotation_utils",
        ":type_system_tracer",
        "//xls/common:xls_gunit_main",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "parametric_struct_instantiator",
    hdrs = ["parametric_struct_instantiator.h"],
    deps = [
        ":inference_table",
        ":type_annotation_utils",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "type_annotation_resolver",
    srcs = ["type_annotation_resolver.cc"],
    hdrs = ["type_annotation_resolver.h"],
    deps = [
        ":evaluator",
        ":inference_table",
        ":parametric_struct_instantiator",
        ":type_annotation_utils",
        ":type_system_tracer",
        ":unify_type_annotations",
        "//xls/common:casts",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        ":type_system_test_utils",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/type_system:typecheck_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)
