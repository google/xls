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
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "concrete_type",
    srcs = ["concrete_type.cc"],
    hdrs = ["concrete_type.h"],
    deps = [
        ":parametric_expression",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
    ],
)

cc_test(
    name = "concrete_type_test",
    srcs = ["concrete_type_test.cc"],
    deps = [
        ":concrete_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "deduce_ctx",
    srcs = ["deduce_ctx.cc"],
    hdrs = ["deduce_ctx.h"],
    deps = [
        ":concrete_type",
        ":type_and_bindings",
        ":type_mismatch_error_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "//xls/common:string_to_int",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:warning_collector",
    ],
)

cc_library(
    name = "deduce",
    srcs = ["deduce.cc"],
    hdrs = ["deduce.h"],
    deps = [
        ":concrete_type",
        ":concrete_type_zero_value",
        ":deduce_ctx",
        ":parametric_constraint",
        ":parametric_instantiator",
        ":type_and_bindings",
        ":unwrap_meta_type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
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
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:token_utils",
    ],
)

cc_library(
    name = "parametric_constraint",
    srcs = ["parametric_constraint.cc"],
    hdrs = ["parametric_constraint.h"],
    deps = [
        ":concrete_type",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "parametric_instantiator_internal",
    srcs = ["parametric_instantiator_internal.cc"],
    hdrs = ["parametric_instantiator_internal.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        ":parametric_bind",
        ":parametric_constraint",
        ":unwrap_meta_type",
        "@com_google_absl//absl/strings",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "parametric_instantiator",
    srcs = ["parametric_instantiator.cc"],
    hdrs = ["parametric_instantiator.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        ":parametric_constraint",
        ":parametric_instantiator_internal",
        ":type_and_bindings",
        ":type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
    ],
)

cc_library(
    name = "type_mismatch_error_data",
    hdrs = ["type_mismatch_error_data.h"],
    deps = [
        ":concrete_type",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "maybe_explain_error",
    srcs = ["maybe_explain_error.cc"],
    hdrs = ["maybe_explain_error.h"],
    deps = [
        ":type_mismatch_error_data",
        "@com_google_absl//absl/status",
        "//xls/dslx:errors",
    ],
)

cc_library(
    name = "typecheck",
    srcs = ["typecheck.cc"],
    hdrs = ["typecheck.h"],
    deps = [
        ":deduce",
        ":deduce_ctx",
        ":maybe_explain_error",
        ":parametric_env",
        ":unwrap_meta_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:dslx_builtins",
        "//xls/dslx:errors",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtins_metadata",
        "@com_github_google_re2//:re2",
    ],
)

cc_library(
    name = "typecheck_test_helpers",
    testonly = True,
    srcs = ["typecheck_test_helpers.cc"],
    hdrs = ["typecheck_test_helpers.h"],
    deps = [
        ":type_info_to_proto",
        ":typecheck",
        "@com_google_absl//absl/status",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
    ],
)

cc_test(
    name = "typecheck_test",
    srcs = ["typecheck_test.cc"],
    deps = [
        ":typecheck",
        ":typecheck_test_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "proc_typecheck_test",
    srcs = ["proc_typecheck_test.cc"],
    deps = [
        ":typecheck",
        ":typecheck_test_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parametric_expression",
    srcs = ["parametric_expression.cc"],
    hdrs = ["parametric_expression.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:pos",
    ],
)

cc_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.cc"],
    deps = [
        ":parametric_expression",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "concrete_type_zero_value",
    srcs = ["concrete_type_zero_value.cc"],
    hdrs = ["concrete_type_zero_value.h"],
    deps = [
        ":concrete_type",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
    ],
)

cc_library(
    name = "parametric_env",
    srcs = ["parametric_env.cc"],
    hdrs = ["parametric_env.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/dslx:interp_value",
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
        ":type_info",
        ":type_info_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:proto_adaptor_utils",
        "//xls/dslx:import_data",
    ],
)

cc_test(
    name = "type_info_to_proto_test",
    srcs = ["type_info_to_proto_test.cc"],
    data = glob(["testdata/*.txt"]),
    deps = [
        ":type_info_to_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "type_and_bindings",
    hdrs = ["type_and_bindings.h"],
    deps = [
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:parametric_env",
    ],
)

cc_library(
    name = "type_info",
    srcs = ["type_info.cc"],
    hdrs = ["type_info.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common/status:ret_check",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:parametric_env",
    ],
)

cc_test(
    name = "type_info_test",
    srcs = ["type_info_test.cc"],
    deps = [
        ":type_info",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parametric_bind",
    srcs = ["parametric_bind.cc"],
    hdrs = ["parametric_bind.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
        "//xls/dslx/frontend:pos",
    ],
)

cc_test(
    name = "parametric_bind_test",
    srcs = ["parametric_bind_test.cc"],
    deps = [
        ":parametric_bind",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "unwrap_meta_type",
    srcs = ["unwrap_meta_type.cc"],
    hdrs = ["unwrap_meta_type.h"],
    deps = [
        ":concrete_type",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
    ],
)

# -- binary for invoking type checking subsystem

cc_binary(
    name = "typecheck_main",
    srcs = ["typecheck_main.cc"],
    deps = [
        ":type_info_to_proto",
        ":typecheck",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common:runfiles",
    ],
)
