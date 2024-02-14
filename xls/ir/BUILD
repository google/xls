# Copyright 2020 The XLS Authors
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

load("@xls_pip_deps//:requirements.bzl", "requirement")

# pytype library only
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bit_push_buffer",
    hdrs = ["bit_push_buffer.h"],
    deps = [
        "//xls/common:math_util",
    ],
)

cc_test(
    name = "bit_push_buffer_test",
    srcs = ["bit_push_buffer_test.cc"],
    deps = [
        ":bit_push_buffer",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "bits",
    srcs = ["bits.cc"],
    hdrs = ["bits.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":bit_push_buffer",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "benchmark_support",
    testonly = True,
    srcs = ["benchmark_support.cc"],
    hdrs = ["benchmark_support.h"],
    deps = [
        ":bits",
        ":function_builder",
        ":ir",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "benchmark_support_test",
    srcs = ["benchmark_support_test.cc"],
    deps = [
        ":benchmark_support",
        ":bits",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "foreign_function",
    srcs = ["foreign_function.cc"],
    hdrs = ["foreign_function.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":code_template",
        ":foreign_function_data_cc_proto",
        ":format_preference",
        ":value",
        "//xls/common/logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "foreign_function_test",
    srcs = ["foreign_function_test.cc"],
    deps = [
        ":bits",
        ":foreign_function",
        ":foreign_function_data_cc_proto",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "code_template",
    srcs = ["code_template.cc"],
    hdrs = ["code_template.h"],
    deps = [
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "code_template_test",
    srcs = ["code_template_test.cc"],
    deps = [
        ":code_template",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "foreign_function_data_proto",
    srcs = ["foreign_function_data.proto"],
)

cc_proto_library(
    name = "foreign_function_data_cc_proto",
    deps = [":foreign_function_data_proto"],
)

cc_library(
    name = "caret",
    srcs = ["caret.cc"],
    hdrs = ["caret.h"],
    deps = [
        ":source_location",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "interval",
    srcs = ["interval.cc"],
    hdrs = ["interval.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":format_preference",
        "//xls/common/logging",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "interval_test",
    size = "small",
    srcs = [
        "interval_test.cc",
    ],
    deps = [
        ":bits",
        ":bits_ops",
        ":interval",
        ":interval_test_utils",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_library(
    name = "interval_ops",
    srcs = ["interval_ops.cc"],
    hdrs = ["interval_ops.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":interval",
        ":interval_set",
        ":ir",
        ":ternary",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "interval_ops_test",
    size = "small",
    srcs = [
        "interval_ops_test.cc",
    ],
    deps = [
        ":bits",
        ":interval",
        ":interval_ops",
        ":interval_set",
        ":ternary",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "interval_set",
    srcs = ["interval_set.cc"],
    hdrs = ["interval_set.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":interval",
        "//xls/common/logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "interval_test_utils",
    testonly = True,
    hdrs = ["interval_test_utils.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":bits_test_utils",
        ":interval",
        "//xls/common/logging",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_library(
    name = "interval_set_test_utils",
    testonly = True,
    hdrs = ["interval_set_test_utils.h"],
    deps = [
        ":interval",
        ":interval_set",
        ":interval_test_utils",
        "//xls/common/logging",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_test(
    name = "interval_set_test",
    size = "small",
    srcs = [
        "interval_set_test.cc",
    ],
    deps = [
        ":bits",
        ":interval",
        ":interval_set",
        ":interval_set_test_utils",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_library(
    name = "ternary",
    srcs = ["ternary.cc"],
    hdrs = ["ternary.h"],
    deps = [
        ":bits",
        "//xls/common/logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "ternary_test",
    size = "small",
    srcs = [
        "ternary_test.cc",
    ],
    deps = [
        ":ternary",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "bits_test_utils",
    testonly = True,
    hdrs = ["bits_test_utils.h"],
    deps = [
        ":bits",
        "//xls/common/logging",
        "//xls/data_structures:inline_bitmap",
        "@com_google_fuzztest//fuzztest",
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
        ":bits_ops",
        ":bits_test_utils",
        ":number_parser",
        ":value",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        ":xls_type_cc_proto",
        "//xls/common:casts",
        "//xls/common/logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "op",
    srcs = ["op.cc"],
    hdrs = ["op.h"],
    deps = [
        ":op_cc_proto",
        "//xls/common/logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_proto_library(
    name = "op_cc_proto",
    deps = [":op_proto"],
)

proto_library(
    name = "op_proto",
    srcs = ["op.proto"],
)

py_proto_library(
    name = "op_py_pb2",
    deps = [":op_proto"],
)

cc_library(
    name = "register",
    srcs = [
        "register.cc",
    ],
    hdrs = ["register.h"],
    deps = [
        ":type",
        ":value",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "events",
    srcs = ["events.cc"],
    hdrs = ["events.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ir",
    srcs = [
        "block.cc",
        "call_graph.cc",
        "dfs_visitor.cc",
        "function.cc",
        "function_base.cc",
        "instantiation.cc",
        "node.cc",
        "node_iterator.cc",
        "nodes.cc",
        "package.cc",
        "proc.cc",
        "proc_instantiation.cc",
        "verifier.cc",
    ],
    hdrs = [
        "block.h",
        "call_graph.h",
        "dfs_visitor.h",
        "function.h",
        "function_base.h",
        "instantiation.h",
        "lsb_or_msb.h",
        "node.h",
        "node_iterator.h",
        "nodes.h",
        "package.h",
        "proc.h",
        "proc_instantiation.h",
        "verifier.h",
    ],
    deps = [
        ":bits",
        ":caret",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":code_template",
        ":foreign_function",
        ":foreign_function_data_cc_proto",
        ":format_strings",
        ":ir_scanner",
        ":name_uniquer",
        ":op",
        ":register",
        ":source_location",
        ":type",
        ":unwrapping_iterator",
        ":value",
        ":value_utils",
        ":xls_type_cc_proto",
        "//xls/common:casts",
        "//xls/common:iterator_range",
        "//xls/common:math_util",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "node_iterator_test",
    srcs = ["node_iterator_test.cc"],
    deps = [
        ":benchmark_support",
        ":bits",
        ":function_builder",
        ":ir",
        ":ir_parser",
        ":ir_test_base",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function_builder",
    srcs = ["function_builder.cc"],
    hdrs = ["function_builder.h"],
    deps = [
        ":channel",
        ":channel_ops",
        ":format_strings",
        ":ir",
        ":op",
        ":register",
        ":source_location",
        ":value",
        "//xls/common:casts",
        "//xls/common:symbolized_stacktrace",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "function_builder_test",
    srcs = ["function_builder_test.cc"],
    deps = [
        ":bits",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":node_util",
        ":source_location",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "source_location",
    hdrs = [
        "fileno.h",
        "source_location.h",
    ],
    deps = [
        "//xls/common:strong_int",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ir_scanner",
    srcs = ["ir_scanner.cc"],
    hdrs = ["ir_scanner.h"],
    deps = [
        ":bits",
        ":number_parser",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "ir_scanner_test",
    srcs = ["ir_scanner_test.cc"],
    deps = [
        ":ir_scanner",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "ir_parser",
    srcs = ["ir_parser.cc"],
    hdrs = ["ir_parser.h"],
    deps = [
        ":bits",
        ":bits_ops",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":code_template",
        ":foreign_function_data_cc_proto",
        ":format_strings",
        ":function_builder",
        ":ir",
        ":ir_scanner",
        ":op",
        ":register",
        ":source_location",
        ":type",
        ":value",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "ir_parser_test",
    size = "small",
    srcs = ["ir_parser_test.cc"],
    deps = [
        "ir_parser",
        ":bits",
        ":bits_ops",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":ir",
        ":op",
        ":value",
        "//xls/common:casts",
        "//xls/common:source_location",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "package_test",
    size = "small",
    srcs = ["package_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":ir_test_base",
        ":type",
        ":value",
        ":xls_type_cc_proto",
        "//xls/common:casts",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "node_util_test",
    size = "small",
    srcs = ["node_util_test.cc"],
    data = glob(["testdata/*.ir"]),
    deps = [
        ":bits",
        ":channel",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":ir_test_base",
        ":node_util",
        ":op",
        ":source_location",
        ":value",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ir_test_base",
    testonly = True,
    srcs = ["ir_test_base.cc"],
    hdrs = ["ir_test_base.h"],
    deps = [
        ":bits",
        ":ir",
        ":ir_parser",
        ":op",
        ":value",
        ":value_test_util",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/common:source_location",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/interpreter:ir_interpreter",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/scheduling:pipeline_schedule",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dfs_visitor_test",
    srcs = ["dfs_visitor_test.cc"],
    deps = [
        ":ir",
        ":ir_test_base",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "verifier_test",
    srcs = ["verifier_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "value",
    srcs = ["value.cc"],
    hdrs = ["value.h"],
    deps = [
        ":bit_push_buffer",
        ":bits",
        ":bits_ops",
        ":format_preference",
        ":xls_type_cc_proto",
        ":xls_value_cc_proto",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "value_utils",
    srcs = ["value_utils.cc"],
    hdrs = ["value_utils.h"],
    deps = [
        ":type",
        ":value",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "value_test",
    srcs = ["value_test.cc"],
    deps = [
        ":bits",
        ":ir_parser",
        ":value",
        ":xls_value_cc_proto",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "node_test",
    srcs = ["node_test.cc"],
    deps = [
        ":bits",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":op",
        ":source_location",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "function_test",
    srcs = ["function_test.cc"],
    deps = [
        ":bits",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":node_util",
        ":op",
        ":source_location",
        ":type",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "ir_test_base_test",
    srcs = ["ir_test_base_test.cc"],
    deps = [
        ":ir_test_base",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_test(
    name = "value_test_util_test",
    srcs = ["value_test_util_test.cc"],
    deps = [
        ":value",
        ":value_test_util",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":type",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "format_preference",
    srcs = ["format_preference.cc"],
    hdrs = ["format_preference.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "format_preference_test",
    srcs = ["format_preference_test.cc"],
    deps = [
        ":format_preference",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_library(
    name = "format_strings",
    srcs = ["format_strings.cc"],
    hdrs = ["format_strings.h"],
    deps = [
        ":format_preference",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "format_strings_test",
    srcs = ["format_strings_test.cc"],
    deps = [
        ":format_strings",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_library(
    name = "value_test_util",
    testonly = True,
    srcs = ["value_test_util.cc"],
    hdrs = ["value_test_util.h"],
    deps = [
        ":value",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

filegroup(
    name = "templates",
    srcs = glob(["*.tmpl"]),
)

# op_specification is very slow to type check so don't use a type checking
# target. Types are checked separately in the build test
# op_specification_build_test.
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
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
    ],
)

py_library(
    name = "op_specification_typechecked",
    srcs = [
        "op_specification.py",
    ],
    srcs_version = "PY2AND3",
)

build_test(
    name = "op_specification_build_test",
    targets = ["op_specification_typechecked"],
)

genrule(
    name = "op_header",
    srcs = ["op_header.tmpl"],
    outs = ["op.h"],
    cmd = "$(location :render_specification_against_template) $(location :op_header.tmpl)" +
          " | $(location @llvm_toolchain//:clang-format)" +
          " > $(OUTS)",
    tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:clang-format",
    ],
)

genrule(
    name = "gen_op_proto",
    srcs = ["op_proto.tmpl"],
    outs = ["op.generated.proto"],
    cmd = "$(location :render_specification_against_template) $(location :op_proto.tmpl) > $(OUTS)",
    tools = [
        ":render_specification_against_template",
    ],
)

# TODO(taktoa): remove this once py_proto_library supports genrule deps
sh_test(
    name = "check_generated_op_proto",
    size = "small",
    srcs = ["check_generated_op_proto.sh"],
    args = [
        "$(location :op.generated.proto)",
        "$(location :op.proto)",
    ],
    data = [
        ":op.generated.proto",
        ":op.proto",
    ],
)

genrule(
    name = "op_source",
    srcs = ["op_source.tmpl"],
    outs = ["op.cc"],
    cmd = "$(location :render_specification_against_template) $(location :op_source.tmpl)" +
          " | $(location @llvm_toolchain//:clang-format)" +
          " > $(OUTS)",
    tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:clang-format",
    ],
)

genrule(
    name = "nodes_header",
    srcs = ["nodes_header.tmpl"],
    outs = ["nodes.h"],
    cmd = "$(location :render_specification_against_template) $(location :nodes_header.tmpl)" +
          " | $(location @llvm_toolchain//:clang-format)" +
          " > $(OUTS)",
    tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:clang-format",
    ],
)

genrule(
    name = "nodes_source",
    srcs = ["nodes_source.tmpl"],
    outs = ["nodes.cc"],
    cmd = "$(location :render_specification_against_template) $(location :nodes_source.tmpl)" +
          " | $(location @llvm_toolchain//:clang-format)" +
          " > $(OUTS)",
    tools = [
        ":render_specification_against_template",
        "@llvm_toolchain//:clang-format",
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
        "//xls/common:string_to_int",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "bits_ops",
    srcs = ["bits_ops.cc"],
    hdrs = ["bits_ops.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":big_int",
        ":bits",
        ":format_preference",
        ":op",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "node_util",
    srcs = ["node_util.cc"],
    hdrs = ["node_util.h"],
    deps = [
        ":bits",
        ":channel",
        ":ir",
        ":op",
        ":source_location",
        ":type",
        ":value",
        ":value_utils",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "big_int",
    srcs = ["big_int.cc"],
    hdrs = ["big_int.h"],
    deps = [
        ":bits",
        "//xls/common:bits_util",
        "//xls/common/logging",
        "@boringssl//:crypto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "number_parser_test",
    srcs = ["number_parser_test.cc"],
    deps = [
        ":bits_ops",
        ":number_parser",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_test(
    name = "big_int_test",
    srcs = ["big_int_test.cc"],
    deps = [
        ":big_int",
        ":bits",
        ":bits_ops",
        ":number_parser",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_test(
    name = "bits_ops_test",
    srcs = [
        "bits_ops_test.cc",
    ],
    deps = [
        ":bits",
        ":bits_ops",
        ":bits_test_utils",
        ":format_preference",
        ":number_parser",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_test(
    name = "caret_test",
    srcs = ["caret_test.cc"],
    deps = [
        ":caret",
        ":ir",
        ":source_location",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/strings:str_format",
    ],
)

proto_library(
    name = "xls_value_proto",
    srcs = ["xls_value.proto"],
    visibility = ["//xls:xls_users"],
)

cc_proto_library(
    name = "xls_value_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":xls_value_proto"],
)

py_proto_library(
    name = "xls_value_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":xls_value_proto"],
)

proto_library(
    name = "xls_type_proto",
    srcs = ["xls_type.proto"],
)

cc_proto_library(
    name = "xls_type_cc_proto",
    deps = [":xls_type_proto"],
)

py_proto_library(
    name = "xls_type_py_pb2",
    deps = [":xls_type_proto"],
)

proto_library(
    name = "channel_proto",
    srcs = ["channel.proto"],
)

cc_proto_library(
    name = "channel_cc_proto",
    deps = [":channel_proto"],
)

py_proto_library(
    name = "channel_py_pb2",
    deps = [":channel_proto"],
)

cc_library(
    name = "keyword_args",
    srcs = ["keyword_args.cc"],
    hdrs = ["keyword_args.h"],
    deps = [
        ":ir",
        ":value",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
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
        ":bits",
        ":channel",
        ":format_preference",
        ":ir",
        ":ir_parser",
        ":op",
        ":type",
        ":value",
        "//xls/common/logging",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "abstract_evaluator",
    hdrs = ["abstract_evaluator.h"],
    deps = [
        ":bits",
        ":bits_ops",
        "//xls/common:math_util",
        "//xls/common/logging",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "abstract_node_evaluator",
    hdrs = ["abstract_node_evaluator.h"],
    deps = [
        ":ir",
        ":op",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "value_view",
    hdrs = ["value_view.h"],
    deps = [
        ":ir",
        ":type",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common/logging",
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_library(
    name = "value_view_utils",
    hdrs = ["value_view_utils.h"],
    deps = [
        ":value_view",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "channel_ops",
    srcs = ["channel_ops.cc"],
    hdrs = ["channel_ops.h"],
    deps = [
        "//xls/common/logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "channel",
    srcs = ["channel.cc"],
    hdrs = ["channel.h"],
    deps = [
        ":channel_cc_proto",
        ":channel_ops",
        ":type",
        ":value",
        ":value_utils",
        "//xls/common:casts",
        "//xls/common/logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "name_uniquer",
    srcs = ["name_uniquer.cc"],
    hdrs = ["name_uniquer.h"],
    deps = [
        "//xls/common/logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "value_view_test",
    srcs = ["value_view_test.cc"],
    deps = [
        ":bits",
        ":bits_ops",
        ":value_utils",
        ":value_view",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "abstract_evaluator_test",
    srcs = ["abstract_evaluator_test.cc"],
    shard_count = 10,
    deps = [
        ":abstract_evaluator",
        ":big_int",
        ":bits",
        ":bits_ops",
        ":bits_test_utils",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_test(
    name = "ir_matcher_test",
    srcs = ["ir_matcher_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":register",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "value_utils_test",
    srcs = ["value_utils_test.cc"],
    deps = [
        ":bits_ops",
        ":ir",
        ":ir_parser",
        ":value",
        ":value_utils",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_test(
    name = "proc_test",
    srcs = ["proc_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":ir_test_base",
        ":op",
        ":source_location",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "channel_test",
    srcs = ["channel_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":ir",
        ":ir_parser",
        ":value",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "name_uniquer_test",
    srcs = ["name_uniquer_test.cc"],
    deps = [
        ":name_uniquer",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_test(
    name = "block_test",
    srcs = ["block_test.cc"],
    deps = [
        ":bits",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":register",
        ":source_location",
        ":type",
        ":value",
        "//xls/common:visitor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "value_conversion_utils",
    srcs = ["value_conversion_utils.cc"],
    hdrs = ["value_conversion_utils.h"],
    deps = [
        ":bits",
        ":type",
        ":value",
        ":value_utils",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "elaboration",
    srcs = ["elaboration.cc"],
    hdrs = ["elaboration.h"],
    deps = [
        ":channel",
        ":channel_ops",
        ":ir",
        ":value",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "proc_conversion",
    srcs = ["proc_conversion.cc"],
    hdrs = ["proc_conversion.h"],
    deps = [
        ":channel",
        ":channel_ops",
        ":ir",
        ":node_util",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "proc_conversion_test",
    srcs = ["proc_conversion_test.cc"],
    deps = [
        ":channel",
        ":channel_ops",
        ":elaboration",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":proc_conversion",
        ":type",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "elaboration_test",
    srcs = ["elaboration_test.cc"],
    deps = [
        ":channel",
        ":channel_ops",
        ":elaboration",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "call_graph_test",
    srcs = ["call_graph_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_test_base",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_test(
    name = "value_conversion_utils_test",
    srcs = ["value_conversion_utils_test.cc"],
    deps = [
        ":ir",
        ":type",
        ":value",
        ":value_conversion_utils",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

proto_library(
    name = "ram_rewrite_proto",
    srcs = ["ram_rewrite.proto"],
)

cc_proto_library(
    name = "ram_rewrite_cc_proto",
    deps = [":ram_rewrite_proto"],
)

cc_library(
    name = "fuzz_type_domain",
    testonly = True,
    srcs = ["fuzz_type_domain.cc"],
    hdrs = ["fuzz_type_domain.h"],
    deps = [
        ":xls_type_cc_proto",
        "//xls/common:math_util",
        "//xls/common/logging",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_test(
    name = "fuzz_type_domain_test",
    srcs = ["fuzz_type_domain_test.cc"],
    deps = [
        ":fuzz_type_domain",
        ":xls_type_cc_proto",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)
