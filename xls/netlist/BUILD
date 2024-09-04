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

# pytype tests are present in this file
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

exports_files(["fake_cell_library.textproto"])

cc_library(
    name = "netlist",
    hdrs = [
        "netlist.h",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":cell_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:bits_util",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "fake_cell_library",
    testonly = True,
    srcs = ["fake_cell_library.cc"],
    hdrs = ["fake_cell_library.h"],
    data = [":fake_cell_library.textproto"],
    deps = [
        ":cell_library",
        ":netlist_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "interpreter",
    hdrs = [
        "interpreter.h",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":cell_library",
        ":function_parser",
        ":netlist",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//xls/common:thread",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.cc"],
    deps = [
        ":cell_library",
        ":fake_cell_library",
        ":function_extractor",
        ":interpreter",
        ":lib_parser",
        ":netlist",
        ":netlist_cc_proto",
        ":netlist_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "netlist_parser",
    srcs = ["netlist_parser.cc"],
    hdrs = ["netlist_parser.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":cell_library",
        ":netlist",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:string_to_int",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:inline_bitmap",
        "//xls/ir:bits",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "netlist_parser_test",
    srcs = ["netlist_parser_test.cc"],
    deps = [
        ":cell_library",
        ":fake_cell_library",
        ":netlist",
        ":netlist_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

proto_library(
    name = "netlist_proto",
    srcs = ["netlist.proto"],
)

cc_proto_library(
    name = "netlist_cc_proto",
    deps = [":netlist_proto"],
)

cc_library(
    name = "cell_library",
    srcs = ["cell_library.cc"],
    hdrs = ["cell_library.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":netlist_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "cell_library_test",
    srcs = ["cell_library_test.cc"],
    deps = [
        ":cell_library",
        ":netlist_cc_proto",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "parse_netlist_main",
    srcs = ["parse_netlist_main.cc"],
    deps = [
        ":cell_library",
        ":find_logic_clouds",
        ":netlist",
        ":netlist_cc_proto",
        ":netlist_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
    ],
)

sh_test(
    name = "parse_netlist_main_test",
    srcs = ["parse_netlist_main_test.sh"],
    data = [
        ":parse_netlist_main",
    ],
)

cc_library(
    name = "find_logic_clouds",
    srcs = ["find_logic_clouds.cc"],
    hdrs = ["find_logic_clouds.h"],
    deps = [
        ":cell_library",
        ":netlist",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/data_structures:union_find",
    ],
)

cc_test(
    name = "find_logic_clouds_test",
    srcs = ["find_logic_clouds_test.cc"],
    deps = [
        ":cell_library",
        ":fake_cell_library",
        ":find_logic_clouds",
        ":netlist",
        ":netlist_parser",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "logical_effort",
    srcs = ["logical_effort.cc"],
    hdrs = ["logical_effort.h"],
    deps = [
        ":cell_library",
        ":netlist",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "logical_effort_test",
    srcs = ["logical_effort_test.cc"],
    deps = [
        ":cell_library",
        ":fake_cell_library",
        ":logical_effort",
        ":netlist",
        ":netlist_parser",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "lib_parser",
    srcs = ["lib_parser.cc"],
    hdrs = ["lib_parser.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "lib_parser_test",
    srcs = ["lib_parser_test.cc"],
    deps = [
        ":lib_parser",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "function_extractor",
    srcs = ["function_extractor.cc"],
    hdrs = ["function_extractor.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":lib_parser",
        ":netlist_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_binary(
    name = "function_extractor_main",
    srcs = ["function_extractor_main.cc"],
    deps = [
        ":function_extractor",
        ":lib_parser",
        ":netlist_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "function_extractor_test",
    srcs = ["function_extractor_test.cc"],
    deps = [
        ":function_extractor",
        ":lib_parser",
        ":netlist_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "function_parser",
    srcs = ["function_parser.cc"],
    hdrs = ["function_parser.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "function_parser_test",
    srcs = ["function_parser_test.cc"],
    deps = [
        ":function_parser",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "netlist_interpreter_main",
    srcs = ["netlist_interpreter_main.cc"],
    deps = [
        ":cell_library",
        ":function_extractor",
        ":interpreter",
        ":lib_parser",
        ":netlist_cc_proto",
        ":netlist_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:flattening",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/netlist",
    ],
)

netlist_interpreter_main_test_data = [
    "testdata/isqrt.v",
    "testdata/ifte.v",
    "testdata/simple_cell.lib",
]

py_test(
    name = "netlist_interpreter_main_test",
    srcs = ["netlist_interpreter_main_test.py"],
    data = glob(netlist_interpreter_main_test_data) + [
        ":netlist_interpreter_main",
    ],
    main = "netlist_interpreter_main_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)
