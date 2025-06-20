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
    testonly = 1,
    srcs = ["ir_fuzz_domain.cc"],
    hdrs = ["ir_fuzz_domain.h"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_builder",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:ret_check",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:verifier",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "ir_fuzz_test",
    srcs = ["ir_fuzz_test.cc"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_domain",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_test_base",
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
        ":ir_fuzz_visitor",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
    ],
)

cc_library(
    name = "stringify_program_pass",
    srcs = ["stringify_program_pass.cc"],
    hdrs = ["stringify_program_pass.h"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_visitor",
    ],
)

cc_library(
    name = "combine_stack",
    srcs = ["combine_stack.cc"],
    hdrs = ["combine_stack.h"],
    deps = [
        ":fuzz_program_cc_proto",
        "//xls/ir:function_builder",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ir_fuzz_builder",
    srcs = ["ir_fuzz_builder.cc"],
    hdrs = ["ir_fuzz_builder.h"],
    deps = [
        ":combine_stack",
        ":fuzz_program_cc_proto",
        ":gen_ir_nodes_pass",
        ":stringify_program_pass",
        "//xls/ir",
        "//xls/ir:function_builder",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "ir_fuzz_visitor",
    srcs = ["ir_fuzz_visitor.cc"],
    hdrs = ["ir_fuzz_visitor.h"],
    deps = [":fuzz_program_cc_proto"],
)

cc_test(
    name = "ir_fuzz_builder_test",
    srcs = ["ir_fuzz_builder_test.cc"],
    deps = [
        ":fuzz_program_cc_proto",
        ":ir_fuzz_builder",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:verifier",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)
