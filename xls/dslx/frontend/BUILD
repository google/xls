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

# Front-end types and routines for e.g. parsing/scanning/AST.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ast_cloner",
    srcs = ["ast_cloner.cc"],
    hdrs = ["ast_cloner.h"],
    deps = [
        ":ast",
        ":ast_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "ast_cloner_test",
    srcs = ["ast_cloner_test.cc"],
    deps = [
        ":ast",
        ":ast_cloner",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:parse_and_typecheck",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "token_parser",
    srcs = ["token_parser.cc"],
    hdrs = ["token_parser.h"],
    deps = [
        ":bindings",
        ":scanner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "parser",
    srcs = ["parser.cc"],
    hdrs = ["parser.h"],
    deps = [
        ":ast",
        ":ast_cloner",
        ":ast_utils",
        ":bindings",
        ":builtins_metadata",
        ":scanner",
        ":token_parser",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common:strong_int",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_builder",
        "//xls/common/status:status_macros",
        "//xls/ir:name_uniquer",
        "@com_github_google_re2//:re2",
    ],
)

cc_test(
    name = "parser_test",
    srcs = ["parser_test.cc"],
    deps = [
        ":builtins_metadata",
        ":parser",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:command_line_utils",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ast_utils",
    srcs = ["ast_utils.cc"],
    hdrs = ["ast_utils.h"],
    deps = [
        ":ast",
        ":token_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx/type_system:type_info",
    ],
)

cc_test(
    name = "ast_utils_test",
    srcs = ["ast_utils_test.cc"],
    deps = [
        ":ast",
        ":ast_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:import_data",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "token_utils",
    srcs = [
        "scanner_keywords.inc",
        "token_utils.cc",
    ],
    hdrs = ["token_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "scanner",
    srcs = ["scanner.cc"],
    hdrs = [
        "scanner.h",
        "scanner_keywords.inc",
    ],
    deps = [
        ":pos",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "scanner_test",
    srcs = ["scanner_test.cc"],
    deps = [
        ":scanner",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "bindings",
    srcs = ["bindings.cc"],
    hdrs = ["bindings.h"],
    deps = [
        ":ast",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "@com_github_google_re2//:re2",
    ],
)

cc_test(
    name = "bindings_test",
    srcs = ["bindings_test.cc"],
    deps = [
        ":bindings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "builtins_metadata",
    srcs = ["builtins_metadata.cc"],
    hdrs = ["builtins_metadata.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pos",
    srcs = ["pos.cc"],
    hdrs = ["pos.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "@com_github_google_re2//:re2",
    ],
)

cc_test(
    name = "pos_test",
    srcs = ["pos_test.cc"],
    deps = [
        ":pos",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ast_builtin_types",
    hdrs = ["ast_builtin_types.inc"],
)

cc_library(
    name = "ast",
    srcs = ["ast.cc"],
    hdrs = ["ast.h"],
    deps = [
        ":ast_builtin_types",
        ":pos",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:casts",
        "//xls/common:indent",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:foreign_function",
        "//xls/ir:format_strings",
        "//xls/ir:number_parser",
    ],
)

cc_test(
    name = "ast_test",
    srcs = ["ast_test.cc"],
    deps = [
        ":ast",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)
