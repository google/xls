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

# Build rules for XLS-used logic solvers.
package(
    default_visibility = ["//xls:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "z3_ir_translator",
    srcs = ["z3_ir_translator.cc"],
    hdrs = ["z3_ir_translator.h"],
    deps = [
        ":z3_utils",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:statusor",
        "//xls/data_structures:leaf_type_tree",
        "//xls/ir",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:abstract_node_evaluator",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_ir_translator_test",
    srcs = ["z3_ir_translator_test.cc"],
    deps = [
        ":z3_ir_translator",
        ":z3_utils",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "@z3//:api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "z3_lec",
    srcs = ["z3_lec.cc"],
    hdrs = ["z3_lec.h"],
    deps = [
        ":z3_ir_translator",
        ":z3_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "//xls/ir",
        "//xls/netlist",
        "//xls/netlist:z3_translator",
        "@z3//:api",
    ],
)

cc_library(
    name = "z3_utils",
    srcs = ["z3_utils.cc"],
    hdrs = ["z3_utils.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/ir:bits",
        "@z3//:api",
        "@com_google_re2//:re2",
    ],
)

cc_test(
    name = "z3_utils_test",
    srcs = ["z3_utils_test.cc"],
    deps = [
        ":z3_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
