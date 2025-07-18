# Copyright 2025 The XLS Authors
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

# Load proto_library
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ir_fuzz_domain",
    testonly = True,
    srcs = ["ir_fuzz_domain.cc"],
    hdrs = ["ir_fuzz_domain.h"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_builder",
        ":ir_fuzz_helpers",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:ret_check",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/ir:verifier",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ir_fuzz_test",
    srcs = ["ir_fuzz_test.cc"],
    deps = [
        ":ir_fuzz_domain",
        ":ir_fuzz_helpers",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:verifier",
        "@com_google_absl//absl/log",
    ],
)

proto_library(
    name = "fuzz_program_proto",
    srcs = ["fuzz_program.proto"],
)

cc_proto_library(
    name = "fuzz_program_cc_proto",
    deps = [":fuzz_program_proto"],
)

cc_library(
    name = "gen_ir_nodes_pass",
    srcs = ["gen_ir_nodes_pass.cc"],
    hdrs = ["gen_ir_nodes_pass.h"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_helpers",
        ":ir_fuzz_visitor",
        ":ir_node_context_list",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_library(
    name = "combine_context_list",
    srcs = ["combine_context_list.cc"],
    hdrs = ["combine_context_list.h"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_helpers",
        ":ir_node_context_list",
        "//xls/ir:function_builder",
    ],
)

cc_library(
    name = "ir_fuzz_builder",
    srcs = ["ir_fuzz_builder.cc"],
    hdrs = ["ir_fuzz_builder.h"],
    deps = [
        ":combine_context_list",
        ":fuzz_program_cc_proto",
        ":gen_ir_nodes_pass",
        ":ir_node_context_list",
        "//xls/ir",
        "//xls/ir:function_builder",
        "@com_google_absl//absl/log",
    ],
)

cc_test(
    name = "ir_fuzz_builder_test",
    srcs = ["ir_fuzz_builder_test.cc"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_builder",
        ":ir_fuzz_helpers",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:verifier",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ir_fuzz_visitor",
    srcs = ["ir_fuzz_visitor.cc"],
    hdrs = ["ir_fuzz_visitor.h"],
    deps = [":fuzz_program_cc_proto"],
)

cc_library(
    name = "ir_fuzz_helpers",
    srcs = ["ir_fuzz_helpers.cc"],
    hdrs = ["ir_fuzz_helpers.h"],
    deps = [
        ":fuzz_program_cc_proto",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ir_fuzz_helpers_test",
    srcs = ["ir_fuzz_helpers_test.cc"],
    deps = [
        ":ir_fuzz_helpers",
        "//xls/common:xls_gunit_main",
        "//xls/ir:bits",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ir_fuzz_test_library",
    testonly = True,
    srcs = ["ir_fuzz_test_library.cc"],
    hdrs = ["ir_fuzz_test_library.h"],
    deps = [
        ":ir_fuzz_domain",
        ":ir_fuzz_helpers",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/ir:verifier",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ir_node_context_list",
    srcs = ["ir_node_context_list.cc"],
    hdrs = ["ir_node_context_list.h"],
    deps = [
        ":ir_fuzz_helpers",
        "//xls/ir",
        "//xls/ir:function_builder",
    ],
)
