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

# Build rules for XLS-used logic solvers.
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
    name = "z3_op_translator",
    srcs = ["z3_op_translator.cc"],
    hdrs = ["z3_op_translator.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_op_translator_test",
    srcs = ["z3_op_translator_test.cc"],
    deps = [
        ":z3_op_translator",
        ":z3_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "@com_google_googletest//:gtest",
        "@z3//:api",
    ],
)

cc_library(
    name = "z3_ir_equivalence",
    srcs = ["z3_ir_equivalence.cc"],
    hdrs = ["z3_ir_equivalence.h"],
    deps = [
        ":z3_ir_translator",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "z3_ir_translator_matchers",
    testonly = True,
    hdrs = ["z3_ir_translator_matchers.h"],
    deps = [
        ":z3_ir_translator",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "z3_ir_translator_matchers_test",
    srcs = ["z3_ir_translator_matchers_test.cc"],
    deps = [
        ":z3_ir_translator",
        ":z3_ir_translator_matchers",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "z3_ir_equivalence_test",
    srcs = ["z3_ir_equivalence_test.cc"],
    deps = [
        ":z3_ir_equivalence",
        ":z3_ir_equivalence_testutils",
        ":z3_ir_translator",
        ":z3_ir_translator_matchers",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "z3_ir_equivalence_testutils",
    testonly = True,
    srcs = ["z3_ir_equivalence_testutils.cc"],
    hdrs = ["z3_ir_equivalence_testutils.h"],
    deps = [
        ":z3_ir_equivalence",
        ":z3_ir_translator",
        ":z3_ir_translator_matchers",
        "//xls/common:source_location",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:proc_testutils",
        "//xls/ir:value",
        "//xls/passes:next_node_modernize_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "z3_ir_translator",
    srcs = ["z3_ir_translator.cc"],
    hdrs = ["z3_ir_translator.h"],
    deps = [
        ":z3_op_translator",
        ":z3_utils",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "//xls/ir",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:abstract_node_evaluator",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_ir_translator_test",
    srcs = ["z3_ir_translator_test.cc"],
    deps = [
        ":z3_ir_translator",
        ":z3_ir_translator_matchers",
        ":z3_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@z3//:api",
    ],
)

cc_library(
    name = "z3_lec",
    srcs = ["z3_lec.cc"],
    hdrs = ["z3_lec.h"],
    deps = [
        ":z3_ir_translator",
        ":z3_netlist_translator",
        ":z3_utils",
        "//xls/codegen/vast",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:node_util",
        "//xls/netlist",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_lec_test",
    srcs = ["z3_lec_test.cc"],
    deps = [
        ":z3_lec",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:fake_cell_library",
        "//xls/netlist:netlist_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "z3_netlist_translator",
    srcs = ["z3_netlist_translator.cc"],
    hdrs = ["z3_netlist_translator.h"],
    deps = [
        ":z3_utils",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:function_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_netlist_translator_test",
    srcs = ["z3_netlist_translator_test.cc"],
    deps = [
        ":z3_netlist_translator",
        ":z3_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:fake_cell_library",
        "//xls/netlist:netlist_parser",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@z3//:api",
    ],
)

cc_library(
    name = "z3_utils",
    srcs = ["z3_utils.cc"],
    hdrs = ["z3_utils.h"],
    deps = [
        "//xls/common:source_location",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_utils_test",
    srcs = ["z3_utils_test.cc"],
    deps = [
        ":z3_op_translator",
        ":z3_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@z3//:api",
    ],
)

cc_binary(
    name = "solver",
    srcs = ["solver.cc"],
    deps = [
        ":z3_ir_translator",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "smtlib_emitter_main",
    srcs = ["smtlib_emitter_main.cc"],
    deps = [
        ":z3_ir_translator",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@z3//:api",
    ],
)
