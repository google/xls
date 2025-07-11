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

# DSLX formatting library components.

# @rules_cc//cc:cc_library.bzl
# @rules_cc//cc:cc_test.bzl

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
    name = "pretty_print",
    srcs = ["pretty_print.cc"],
    hdrs = ["pretty_print.h"],
    deps = [
        "//xls/common:strong_int",
        "//xls/common:visitor",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:token",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "pretty_print_test",
    srcs = ["pretty_print_test.cc"],
    deps = [
        ":pretty_print",
        "//xls/common:xls_gunit_main",
        "//xls/dslx/frontend:pos",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ast_fmt",
    srcs = ["ast_fmt.cc"],
    hdrs = ["ast_fmt.h"],
    deps = [
        ":comments",
        ":format_disabler",
        ":pretty_print",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_cloner",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/frontend:token",
        "//xls/dslx/frontend:token_utils",
        "//xls/ir:channel",
        "//xls/ir:format_strings",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@cppitertools",
    ],
)

cc_library(
    name = "comments",
    srcs = ["comments.cc"],
    hdrs = ["comments.h"],
    deps = [
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "ast_fmt_test",
    srcs = ["ast_fmt_test.cc"],
    deps = [
        ":ast_fmt",
        ":comments",
        ":pretty_print",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_test_utils",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "fmt_generated_ast_test",
    srcs = ["fmt_generated_ast_test.cc"],
    shard_count = 50,
    deps = [
        ":ast_fmt",
        ":comments",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:scanner",
        "//xls/fuzzer:ast_generator",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "format_disabler",
    srcs = ["format_disabler.cc"],
    hdrs = ["format_disabler.h"],
    deps = [
        ":comments",
        "//xls/common/status:status_macros",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "format_disabler_test",
    srcs = ["format_disabler_test.cc"],
    deps = [
        ":comments",
        ":format_disabler",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)
