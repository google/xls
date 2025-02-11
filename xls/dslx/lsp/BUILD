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

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "lsp_uri",
    srcs = ["lsp_uri.cc"],
    hdrs = ["lsp_uri.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "lsp_type_utils",
    srcs = ["lsp_type_utils.cc"],
    hdrs = ["lsp_type_utils.h"],
    deps = [
        ":lsp_uri",
        "//xls/dslx/frontend:pos",
        "@verible//verible/common/lsp:lsp-file-utils",
        "@verible//verible/common/lsp:lsp-protocol",
    ],
)

cc_library(
    name = "find_definition",
    srcs = ["find_definition.cc"],
    hdrs = ["find_definition.h"],
    deps = [
        "//xls/common:visitor",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:deduce_utils",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "language_server_adapter",
    srcs = ["language_server_adapter.cc"],
    hdrs = ["language_server_adapter.h"],
    deps = [
        ":document_symbols",
        ":find_definition",
        ":import_sensitivity",
        ":lsp_type_utils",
        ":lsp_uri",
        "//xls/common:casts",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:extract_module_name",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/fmt:ast_fmt",
        "//xls/dslx/fmt:comments",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/frontend:comment_data",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@nlohmann_json//:singleheader-json",
        "@verible//verible/common/lsp:lsp-file-utils",
        "@verible//verible/common/lsp:lsp-protocol",
        "@verible//verible/common/lsp:lsp-protocol-enums",
    ],
)

cc_test(
    name = "language_server_adapter_test",
    srcs = ["language_server_adapter_test.cc"],
    deps = [
        ":language_server_adapter",
        ":lsp_uri",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/dslx:default_dslx_stdlib_path",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
        "@verible//verible/common/lsp:lsp-file-utils",
        "@verible//verible/common/lsp:lsp-protocol",
    ],
)

cc_binary(
    name = "dslx_ls",
    srcs = ["dslx_ls.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":language_server_adapter",
        ":lsp_uri",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx:default_dslx_stdlib_path",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:singleheader-json",
        "@verible//verible/common/lsp:json-rpc-dispatcher",
        "@verible//verible/common/lsp:lsp-protocol",
        "@verible//verible/common/lsp:lsp-text-buffer",
        "@verible//verible/common/lsp:message-stream-splitter",
    ],
)

cc_library(
    name = "document_symbols",
    srcs = ["document_symbols.cc"],
    hdrs = ["document_symbols.h"],
    deps = [
        ":lsp_type_utils",
        "//xls/common:visitor",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:variant",
        "@verible//verible/common/lsp:lsp-protocol",
        "@verible//verible/common/lsp:lsp-protocol-enums",
    ],
)

cc_test(
    name = "document_symbols_test",
    srcs = ["document_symbols_test.cc"],
    deps = [
        ":document_symbols",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@googletest//:gtest",
        "@verible//verible/common/lsp:lsp-protocol",
        "@verible//verible/common/lsp:lsp-protocol-enums",
    ],
)

cc_library(
    name = "import_sensitivity",
    srcs = ["import_sensitivity.cc"],
    hdrs = ["import_sensitivity.h"],
    deps = [
        ":lsp_uri",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "import_sensitivity_test",
    srcs = ["import_sensitivity_test.cc"],
    deps = [
        ":import_sensitivity",
        ":lsp_uri",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)
