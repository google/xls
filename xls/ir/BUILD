# Copyright 2020 Google LLC
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

# Compiler data structures for converting high level algorithms into Hardware
# Description Language.

# cc_proto_library is used in this file
load("//xls/build:py_proto_library.bzl", "xls_py_proto_library")
# pytype binary only

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bit_push_buffer",
    hdrs = ["bit_push_buffer.h"],
    deps = [
        "//xls/common:integral_types",
        "//xls/common:math_util",
    ],
)

cc_test(
    name = "bit_push_buffer_test",
    srcs = ["bit_push_buffer_test.cc"],
    deps = [
        ":bit_push_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bits",
    srcs = ["bits.cc"],
    hdrs = ["bits.h"],
    deps = [
        ":bit_push_buffer",
        ":format_preference",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:bits_util",
        "//xls/common:integral_types",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:statusor",
        "//xls/data_structures:inline_bitmap",
    ],
)

cc_library(
    name = "ternary",
    srcs = ["ternary.cc"],
    hdrs = ["ternary.h"],
    deps = [
        ":bits",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "bits_test",
    size = "small",
    srcs = [
        "bits_test.cc",
    ],
    deps = [
        ":bits",
        ":number_parser",
        ":value",
        "//xls/common:math_util",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        ":xls_type_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:casts",
        "//xls/common:integral_types",
        "//xls/common/logging",
        "//xls/common/status:statusor",
    ],
)

cc_library(
    name = "op",
    srcs = ["op.cc"],
    hdrs = ["op.h"],
    deps = [
        ":op_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:integral_types",
        "//xls/common/logging",
        "//xls/common/status:statusor",
    ],
)

proto_library(
    name = "op_proto",
    srcs = ["op.proto"],
)

cc_proto_library(
    name = "op_cc_proto",
    deps = [":op_proto"]
)

cc_library(
    name = "ir",
    srcs = [
        "container_hack.inc",
        "container_hack_undef.inc",
        "dfs_visitor.cc",
        "function.cc",
        "node.cc",
        "node_iterator.cc",
        "nodes.cc",
        "package.cc",
        "verifier.cc",
    ],
    hdrs = [
        "dfs_visitor.h",
        "function.h",
        "lsb_or_msb.h",
        "node.h",
        "node_iterator.h",
        "nodes.h",
        "package.h",
        "verifier.h",
    ],
    deps = [
        ":bits",
        ":op",
        ":source_location",
        ":type",
        ":unwrapping_iterator",
        ":value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:casts",
        "//xls/common:integral_types",
        "//xls/common:iterator_range",
        "//xls/common:math_util",
        "//xls/common:strong_int",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "node_iterator_test",
    srcs = ["node_iterator_test.cc"],
    deps = [
        ":ir",
        ":ir_parser",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "function_builder",
    srcs = ["function_builder.cc"],
    hdrs = ["function_builder.h"],
    deps = [
        ":ir",
        ":source_location",
        ":value",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:symbolized_stacktrace",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "function_builder_test",
    srcs = ["function_builder_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_matcher",
        "@com_google_absl//absl/status",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "source_location",
    hdrs = [
        "fileno.h",
        "source_location.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common:integral_types",
        "//xls/common:strong_int",
    ],
)

cc_library(
    name = "ir_scanner",
    srcs = ["ir_scanner.cc"],
    hdrs = ["ir_scanner.h"],
    deps = [
        ":bits",
        ":number_parser",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:integral_types",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "ir_scanner_test",
    srcs = ["ir_scanner_test.cc"],
    deps = [
        ":ir_scanner",
        "@com_google_absl//absl/memory",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ir_parser",
    srcs = ["ir_parser.cc"],
    hdrs = ["ir_parser.h"],
    deps = [
        ":bits_ops",
        ":function_builder",
        ":ir",
        ":ir_scanner",
        ":number_parser",
        ":op",
        ":source_location",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "ir_parser_test",
    size = "small",
    srcs = ["ir_parser_test.cc"],
    deps = [
        "ir_parser",
        ":bits_ops",
        ":number_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "llvm_ir_jit",
    srcs = ["llvm_ir_jit.cc"],
    hdrs = ["llvm_ir_jit.h"],
    deps = [
        ":ir",
        ":keyword_args",
        ":llvm_ir_runtime",
        ":llvm_type_converter",
        ":type",
        ":value",
        ":value_helpers",
        ":value_view",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:vast",
        "//xls/common:integral_types",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@llvm//:Analysis",
        "@llvm//:Core",
        "@llvm//:ExecutionEngine",
        "@llvm//:IPO",
        "@llvm//:JITLink",  # build_cleaner: keep
        "@llvm//:OrcJIT",
        "@llvm//:Support",
        "@llvm//:Target",
        "@llvm//:X86AsmParser",  # build_cleaner: keep
        "@llvm//:X86CodeGen",  # build_cleaner: keep
    ],
)

cc_library(
    name = "llvm_ir_runtime",
    srcs = ["llvm_ir_runtime.cc"],
    hdrs = ["llvm_ir_runtime.h"],
    deps = [
        ":format_preference",
        ":ir_parser",
        ":llvm_type_converter",
        ":type",
        ":value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:integral_types",
        "@llvm//:Core",
        "@llvm//:OrcJIT",
        "@llvm//:Support",
        "@llvm//:Target",
        "@llvm//:X86AsmParser",  # build_cleaner: keep
        "@llvm//:X86CodeGen",  # build_cleaner: keep
    ],
)

cc_library(
    name = "llvm_type_converter",
    srcs = ["llvm_type_converter.cc"],
    hdrs = ["llvm_type_converter.h"],
    deps = [
        ":ir",
        ":ir_parser",
        ":type",
        ":value",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//xls/common:integral_types",
        "//xls/common/logging",
        "//xls/common/status:statusor",
        "@llvm//:Core",
    ],
)

cc_test(
    name = "llvm_ir_jit_test",
    srcs = ["llvm_ir_jit_test.cc"],
    deps = [
        ":ir_evaluator_test",
        ":llvm_ir_jit",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest_main",
        "@com_google_re2//:re2",
    ],
)

cc_test(
    name = "package_test",
    size = "small",
    srcs = ["package_test.cc"],
    deps = [
        "ir",
        ":ir_test_base",
        ":type",
        ":xls_type_cc_proto",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "node_util_test",
    size = "small",
    srcs = ["node_util_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":node_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ir_evaluator_test",
    testonly = True,
    srcs = ["ir_evaluator_test.cc"],
    hdrs = ["ir_evaluator_test.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":ir",
        ":ir_parser",
        ":ir_test_base",
        ":keyword_args",
        "@com_google_absl//absl/strings",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ir_interpreter_stats",
    srcs = ["ir_interpreter_stats.cc"],
    hdrs = ["ir_interpreter_stats.h"],
    deps = [
        ":ir",
        ":ternary",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "//xls/common:integral_types",
    ],
)

cc_library(
    name = "ir_interpreter",
    srcs = ["ir_interpreter.cc"],
    hdrs = ["ir_interpreter.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":function_builder",
        ":ir",
        ":ir_interpreter_stats",
        ":keyword_args",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_library(
    name = "ir_test_base",
    testonly = True,
    srcs = ["ir_test_base.cc"],
    hdrs = ["ir_test_base.h"],
    deps = [
        ":ir",
        ":ir_interpreter",
        ":ir_parser",
        ":value_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/common:source_location",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/passes:standard_pipeline",
        "//xls/scheduling:pipeline_schedule",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ir_interpreter_test",
    size = "small",
    srcs = ["ir_interpreter_test.cc"],
    deps = [
        ":bits",
        ":bits_ops",
        ":ir",
        ":ir_evaluator_test",
        ":ir_interpreter",
        ":ir_parser",
        ":ir_test_base",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dfs_visitor_test",
    srcs = ["dfs_visitor_test.cc"],
    deps = [
        ":ir",
        ":ir_test_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "verifier_test",
    srcs = ["verifier_test.cc"],
    deps = [
        ":ir",
        ":ir_test_base",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "value",
    srcs = ["value.cc"],
    hdrs = ["value.h"],
    deps = [
        ":bits",
        ":xls_type_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "@com_google_re2//:re2",
    ],
)

cc_library(
    name = "value_helpers",
    srcs = ["value_helpers.cc"],
    hdrs = ["value_helpers.h"],
    deps = [
        ":ir",
        ":type",
        ":value",
        "@com_google_absl//absl/strings",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "value_test",
    srcs = ["value_test.cc"],
    deps = [
        ":bits",
        ":ir",
        ":value",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "node_test",
    srcs = ["node_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_test_base",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "function_test",
    srcs = ["function_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_test_base",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ir_test_base_test",
    srcs = ["ir_test_base_test.cc"],
    deps = [
        ":ir_test_base",
        "@com_google_absl//absl/strings",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "value_test_util_test",
    srcs = ["value_test_util_test.cc"],
    deps = [
        ":value",
        ":value_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "format_preference",
    srcs = ["format_preference.cc"],
    hdrs = ["format_preference.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "value_test_util",
    testonly = True,
    srcs = ["value_test_util.cc"],
    hdrs = ["value_test_util.h"],
    deps = [
        ":value",
        "@com_google_googletest//:gtest",
    ],
)

filegroup(
    name = "templates",
    srcs = glob(["*.tmpl"]),
)

py_binary(
    name = "render_specification_against_template",
    srcs = [
        "op_specification.py",
        "render_specification_against_template.py",
    ],
    data = [
        ":templates",
    ],
    python_version = "PY3",
    deps = [
        "@com_google_absl_py//absl:app",
        "@jinja_archive//:jinja2",
        "//xls/common:runfiles",
    ],
)

genrule(
    name = "op_header",
    outs = ["op.h"],
    cmd = "$(location :render_specification_against_template) xls/ir/op_header.tmpl" +
          " | $(location @llvm_toolchain//:bin/clang-format)" +
          " > $(OUTS)",
    exec_tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:bin/clang-format",
    ],
)

genrule(
    name = "gen_op_proto",
    outs = ["op.proto"],
    cmd = "$(location :render_specification_against_template) xls/ir/op_proto.tmpl" +
          " | $(location @llvm_toolchain//:bin/clang-format)" +
          " > $(OUTS)",
    exec_tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:bin/clang-format",
    ],
)

genrule(
    name = "op_source",
    outs = ["op.cc"],
    cmd = "$(location :render_specification_against_template) xls/ir/op_source.tmpl" +
          " | $(location @llvm_toolchain//:bin/clang-format)" +
          " > $(OUTS)",
    exec_tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:bin/clang-format",
    ],
)

genrule(
    name = "nodes_header",
    outs = ["nodes.h"],
    cmd = "$(location :render_specification_against_template) xls/ir/nodes_header.tmpl" +
          " | $(location @llvm_toolchain//:bin/clang-format)" +
          " > $(OUTS)",
    exec_tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:bin/clang-format",
    ],
)

genrule(
    name = "nodes_source",
    outs = ["nodes.cc"],
    cmd = "$(location :render_specification_against_template) xls/ir/nodes_source.tmpl" +
          " | $(location @llvm_toolchain//:bin/clang-format)" +
          " > $(OUTS)",
    exec_tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:bin/clang-format",
    ],
)

cc_library(
    name = "number_parser",
    srcs = ["number_parser.cc"],
    hdrs = ["number_parser.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":format_preference",
        "@com_google_absl//absl/strings",
        "//xls/common:integral_types",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_library(
    name = "bits_ops",
    srcs = ["bits_ops.cc"],
    hdrs = ["bits_ops.h"],
    deps = [
        ":big_int",
        ":bits",
        ":op",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "node_util",
    srcs = ["node_util.cc"],
    hdrs = ["node_util.h"],
    deps = [
        ":bits_ops",
        ":ir",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_library(
    name = "big_int",
    srcs = ["big_int.cc"],
    hdrs = ["big_int.h"],
    deps = [
        ":bits",
        "//xls/common/logging",
        "//xls/common/status:statusor",
        "@boringssl//:crypto",
    ],
)

cc_test(
    name = "number_parser_test",
    srcs = ["number_parser_test.cc"],
    deps = [
        ":bits_ops",
        ":number_parser",
        "//xls/common:integral_types",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "big_int_test",
    srcs = ["big_int_test.cc"],
    deps = [
        ":big_int",
        ":bits",
        ":number_parser",
        "@com_google_absl//absl/strings",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "bits_ops_test",
    srcs = ["bits_ops_test.cc"],
    deps = [
        ":bits_ops",
        ":number_parser",
        ":value",
        "//xls/common:math_util",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_proto_library(
    name = "xls_type_cc_proto",
    deps = [":xls_type_proto"],
)

proto_library(
    name = "xls_type_proto",
    srcs = ["xls_type.proto"],
)

xls_py_proto_library(
    name = "xls_type_py_pb2",
    srcs = ["xls_type.proto"],
    internal_deps = [":xls_type_proto"],
)

cc_library(
    name = "keyword_args",
    srcs = ["keyword_args.cc"],
    hdrs = ["keyword_args.h"],
    deps = [
        ":ir",
        ":value",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_library(
    name = "unwrapping_iterator",
    hdrs = ["unwrapping_iterator.h"],
)

cc_library(
    name = "ir_matcher",
    testonly = True,
    srcs = ["ir_matcher.cc"],
    hdrs = ["ir_matcher.h"],
    deps = [
        ":format_preference",
        ":ir",
        ":ir_parser",
        ":op",
        ":type",
        ":value",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "abstract_evaluator",
    hdrs = ["abstract_evaluator.h"],
    deps = [
        ":bits",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common:integral_types",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "abstract_node_evaluator",
    hdrs = ["abstract_node_evaluator.h"],
    deps = [
        ":ir",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_library(
    name = "value_view",
    hdrs = ["value_view.h"],
    deps = [
        "//xls/common:integral_types",
        "//xls/common:math_util",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "value_view_helpers",
    hdrs = ["value_view_helpers.h"],
    deps = [
        ":value_view",
        "@com_google_absl//absl/base",
    ],
)

cc_test(
    name = "abstract_evaluator_test",
    srcs = ["abstract_evaluator_test.cc"],
    deps = [
        ":abstract_evaluator",
        ":bits",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ir_matcher_test",
    srcs = ["ir_matcher_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_matcher",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "value_helpers_test",
    srcs = ["value_helpers_test.cc"],
    deps = [
        ":bits_ops",
        ":ir",
        ":value",
        ":value_helpers",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
