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

load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")
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
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
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
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
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
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "code_template",
    srcs = ["code_template.cc"],
    hdrs = ["code_template.h"],
    deps = [
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/strings:str_format",
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
        "@com_google_absl//absl/log:check",
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
        ":bits_test_utils",
        ":interval",
        ":interval_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common:iter_util",
        "//xls/passes:ternary_evaluator",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
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
        ":function_builder",
        ":interval",
        ":interval_ops",
        ":interval_set",
        ":interval_test_utils",
        ":ir",
        ":ir_test_base",
        ":ternary",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_ir_translator_matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common:iterator_range",
        "@com_google_absl//absl/log:check",
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
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/log:check",
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
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ternary",
    srcs = ["ternary.cc"],
    hdrs = ["ternary.h"],
    deps = [
        ":bits",
        "//xls/common:iterator_range",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ternary_test",
    size = "small",
    srcs = [
        "ternary_test.cc",
    ],
    deps = [
        ":bits",
        ":bits_ops",
        ":ternary",
        "//xls/common:iterator_range",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "bits_test_utils",
    testonly = True,
    hdrs = ["bits_test_utils.h"],
    deps = [
        ":bits",
        "//xls/common/fuzzing:fuzztest",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/log:check",
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
        ":format_preference",
        ":number_parser",
        ":value",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        ":xls_type_cc_proto",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "type_manager",
    srcs = ["type_manager.cc"],
    hdrs = ["type_manager.h"],
    deps = [
        ":type",
        ":value",
        ":xls_type_cc_proto",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        ":op_list",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/strings:str_format",
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
        "nodes.cc",
        "package.cc",
        "proc.cc",
        "proc_instantiation.cc",
        "topo_sort.cc",
        "verify_node.cc",
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
        "nodes.h",
        "package.h",
        "proc.h",
        "proc_instantiation.h",
        "topo_sort.h",
        "verify_node.h",
    ],
    deps = [
        ":bits",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":foreign_function_data_cc_proto",
        ":format_strings",
        ":ir_scanner",
        ":name_uniquer",
        ":op",
        ":register",
        ":source_location",
        ":type",
        ":type_manager",
        ":unwrapping_iterator",
        ":value",
        ":value_utils",
        ":xls_type_cc_proto",
        "//xls/common:casts",
        "//xls/common:iterator_range",
        "//xls/common:math_util",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
    name = "nodes_test",
    srcs = ["nodes_test.cc"],
    deps = [
        ":ir",
        ":op",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "op_test",
    srcs = ["op_test.cc"],
    deps = [
        ":op",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    deps = [
        ":block_elaboration",
        ":caret",
        ":channel",
        ":code_template",
        ":ir",
        ":ir_scanner",
        ":op",
        ":proc_elaboration",
        ":register",
        ":source_location",
        ":type",
        ":value_utils",
        "//xls/common:casts",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "topo_sort_test",
    srcs = ["topo_sort_test.cc"],
    deps = [
        ":benchmark_support",
        ":bits",
        ":function_builder",
        ":ir",
        ":ir_parser",
        ":ir_test_base",
        ":op",
        ":source_location",
        ":value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "function_builder",
    srcs = ["function_builder.cc"],
    hdrs = ["function_builder.h"],
    deps = [
        ":bits",
        ":channel",
        ":channel_ops",
        ":foreign_function_data_cc_proto",
        ":format_strings",
        ":ir",
        ":op",
        ":register",
        ":source_location",
        ":type",
        ":value",
        ":verifier",
        "//xls/common:casts",
        "//xls/common:symbolized_stacktrace",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        ":channel_ops",
        ":format_preference",
        ":format_strings",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":node_util",
        ":op",
        ":register",
        ":source_location",
        ":type",
        ":value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ir_scanner",
    srcs = ["ir_scanner.cc"],
    hdrs = ["ir_scanner.h"],
    deps = [
        ":bits",
        ":number_parser",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        ":verifier",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
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
        ":bits",
        ":bits_ops",
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":ir",
        ":ir_parser",
        ":op",
        ":type",
        ":value",
        "//xls/common:casts",
        "//xls/common:source_location",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
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
        ":ternary",
        ":value",
        ":value_builder",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ir_test_base",
    testonly = True,
    srcs = ["ir_test_base.cc"],
    hdrs = ["ir_test_base.h"],
    deps = [
        ":bits",
        ":events",
        ":ir",
        ":ir_parser",
        ":op",
        ":type",
        ":value",
        ":value_test_util",
        ":verifier",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen/vast",
        "//xls/common:source_location",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/interpreter:ir_interpreter",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/simulation:default_verilog_simulator",
        "//xls/simulation:module_simulator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dfs_visitor_test",
    srcs = ["dfs_visitor_test.cc"],
    deps = [
        ":ir",
        ":ir_test_base",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "verifier_test",
    srcs = ["verifier_test.cc"],
    deps = [
        ":channel",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":verifier",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        ":bits",
        ":format_preference",
        ":type",
        ":value",
        "//xls/common:bits_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        ":ir_matcher",
        ":ir_test_base",
        ":op",
        ":source_location",
        ":value",
        ":verifier",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ir_test_base_test",
    srcs = ["ir_test_base_test.cc"],
    deps = [
        ":ir",
        ":ir_test_base",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "value_test_util_test",
    srcs = ["value_test_util_test.cc"],
    deps = [
        ":bits",
        ":value",
        ":value_test_util",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":type",
        ":type_manager",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "format_preference",
    srcs = ["format_preference.cc"],
    hdrs = ["format_preference.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "format_preference_test",
    srcs = ["format_preference_test.cc"],
    deps = [
        ":format_preference",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "format_strings",
    srcs = ["format_strings.cc"],
    hdrs = ["format_strings.h"],
    deps = [
        ":format_preference",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "format_strings_test",
    srcs = ["format_strings_test.cc"],
    deps = [
        ":format_preference",
        ":format_strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/status",
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
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        ":ternary",
        ":type",
        ":value",
        ":value_utils",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "@boringssl//:crypto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "number_parser_test",
    srcs = ["number_parser_test.cc"],
    deps = [
        ":bits",
        ":bits_ops",
        ":format_preference",
        ":number_parser",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "big_int_test",
    srcs = ["big_int_test.cc"],
    deps = [
        ":big_int",
        ":bits",
        ":bits_ops",
        ":format_preference",
        ":number_parser",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/data_structures:inline_bitmap",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "caret_test",
    srcs = ["caret_test.cc"],
    deps = [
        ":caret",
        ":ir",
        ":source_location",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
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
    name = "xls_ir_interface_proto",
    srcs = ["xls_ir_interface.proto"],
    deps = [":xls_type_proto"],
)

cc_proto_library(
    name = "xls_ir_interface_cc_proto",
    deps = [":xls_ir_interface_proto"],
)

py_proto_library(
    name = "xls_ir_interface_py_pb2",
    deps = [":xls_ir_interface_proto"],
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
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "abstract_node_evaluator",
    hdrs = ["abstract_node_evaluator.h"],
    deps = [
        ":bits",
        ":ir",
        ":op",
        ":type",
        ":value",
        ":value_utils",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "abstract_node_evaluator_test",
    srcs = ["abstract_node_evaluator_test.cc"],
    deps = [
        ":abstract_evaluator",
        ":abstract_node_evaluator",
        ":bits",
        ":function_builder",
        ":ir_test_base",
        ":value",
        ":value_builder",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/log",
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
        "@com_google_absl//absl/log",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "value_view_test",
    srcs = ["value_view_test.cc"],
    deps = [
        ":bits",
        ":bits_ops",
        ":value",
        ":value_utils",
        ":value_view",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "value_utils_test",
    srcs = ["value_utils_test.cc"],
    deps = [
        ":bits",
        ":ir",
        ":ir_parser",
        ":type",
        ":value",
        ":value_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/data_structures:leaf_type_tree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
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
        ":ir_matcher",
        ":ir_parser",
        ":type",
        ":value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "name_uniquer_test",
    srcs = ["name_uniquer_test.cc"],
    deps = [
        ":name_uniquer",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "block_test",
    srcs = ["block_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":foreign_function_data_cc_proto",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":register",
        ":source_location",
        ":type",
        ":value",
        "//xls/common:visitor",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@com_google_googletest//:gtest",
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
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "value_builder",
    srcs = ["value_builder.cc"],
    hdrs = ["value_builder.h"],
    deps = [
        ":bits",
        ":value",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "value_builder_test",
    srcs = ["value_builder_test.cc"],
    deps = [
        ":bits",
        ":value",
        ":value_builder",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "elaboration",
    srcs = ["elaboration.cc"],
    hdrs = ["elaboration.h"],
    deps = [
        ":ir",
        "//xls/common:casts",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "block_elaboration",
    srcs = [
        "block_elaboration.cc",
        "elaborated_block_dfs_visitor.cc",
    ],
    hdrs = [
        "block_elaboration.h",
        "elaborated_block_dfs_visitor.h",
    ],
    deps = [
        ":channel",
        ":elaboration",
        ":ir",
        ":op",
        "//xls/common:casts",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "elaborated_block_dfs_visitor_test",
    srcs = ["elaborated_block_dfs_visitor_test.cc"],
    deps = [
        ":bits",
        ":block_elaboration",
        ":function_builder",
        ":ir",
        ":ir_parser",
        ":ir_test_base",
        ":source_location",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_elaboration",
    srcs = ["proc_elaboration.cc"],
    hdrs = ["proc_elaboration.h"],
    deps = [
        ":channel",
        ":channel_cc_proto",
        ":channel_ops",
        ":elaboration",
        ":ir",
        ":value",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
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
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":proc_conversion",
        ":proc_elaboration",
        ":type",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "block_elaboration_test",
    srcs = ["block_elaboration_test.cc"],
    deps = [
        ":bits",
        ":block_elaboration",
        ":channel",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":ir_parser",
        ":ir_test_base",
        ":op",
        ":source_location",
        ":value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "proc_elaboration_test",
    srcs = ["proc_elaboration_test.cc"],
    deps = [
        ":channel",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":proc_elaboration",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "call_graph_test",
    srcs = ["call_graph_test.cc"],
    deps = [
        ":function_builder",
        ":ir",
        ":ir_test_base",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "value_conversion_utils_test",
    srcs = ["value_conversion_utils_test.cc"],
    deps = [
        ":bits",
        ":ir",
        ":type",
        ":value",
        ":value_conversion_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "fuzz_type_domain_test",
    srcs = ["fuzz_type_domain_test.cc"],
    deps = [
        ":fuzz_type_domain",
        ":xls_type_cc_proto",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_testutils",
    srcs = ["proc_testutils.cc"],
    hdrs = ["proc_testutils.h"],
    deps = [
        ":bits",
        ":function_builder",
        ":ir",
        ":source_location",
        ":value",
        ":value_utils",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/passes:dce_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "proc_testutils_test",
    srcs = ["proc_testutils_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":channel_ops",
        ":function_builder",
        ":ir",
        ":ir_test_base",
        ":proc_testutils",
        ":value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/solvers:z3_ir_equivalence",
        "//xls/solvers:z3_ir_equivalence_testutils",
        "//xls/solvers:z3_ir_translator_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "clone_package",
    srcs = ["clone_package.cc"],
    hdrs = ["clone_package.h"],
    deps = [
        ":channel",
        ":ir",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "clone_package_test",
    srcs = ["clone_package_test.cc"],
    deps = [
        ":bits",
        ":channel",
        ":channel_ops",
        ":clone_package",
        ":function_builder",
        ":ir",
        ":ir_matcher",
        ":ir_test_base",
        ":proc_testutils",
        ":value",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/solvers:z3_ir_equivalence",
        "//xls/solvers:z3_ir_translator_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "op_list",
    hdrs = ["op_list.h"],
    visibility = ["//visibility:private"],
)

proto_library(
    name = "xls_op_name_proto",
    srcs = ["xls_op_name.proto"],
)

cc_proto_library(
    name = "xls_op_name_cc_proto",
    deps = [":xls_op_name_proto"],
)

py_proto_library(
    name = "xls_op_name_py_pb2",
    deps = [":xls_op_name_proto"],
)

cc_binary(
    name = "generate_op_name_proto_main",
    srcs = ["generate_op_name_proto_main.cc"],
    deps = [
        ":op_list",
        ":xls_op_name_cc_proto",
        "//xls/common:init_xls",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

genrule_wrapper(
    name = "generate_op_name_representations_proto",
    srcs = [],
    outs = ["op_name_representations.textproto"],
    cmd = "$(location :generate_op_name_proto_main) > $(OUTS)",
    tools = [
        ":generate_op_name_proto_main",
    ],
)

filegroup(
    name = "op_name_representations_proto",
    srcs = ["op_name_representations.textproto"],
    visibility = ["//xls/synthesis:__subpackages__"],
)
