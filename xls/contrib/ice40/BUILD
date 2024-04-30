# Copyright 2024 The XLS Authors
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
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "wrap_io",
    srcs = ["wrap_io.cc"],
    hdrs = ["wrap_io.h"],
    deps = [
        ":io_strategy",
        "//xls/codegen:finite_state_machine",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:source_location",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "wrap_io_test",
    srcs = ["wrap_io_test.cc"],
    data = glob(["testdata/wrap_io_test_*.vtxt"]),
    deps = [
        ":ice40_io_strategy",
        ":null_io_strategy",
        ":wrap_io",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:source_location",
        "//xls/simulation:module_testbench",
        "//xls/simulation:module_testbench_thread",
        "//xls/simulation:verilog_test_base",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ice40_io_strategy",
    srcs = ["ice40_io_strategy.cc"],
    hdrs = ["ice40_io_strategy.h"],
    data = ["//xls/contrib/ice40/uncore_rtl/ice40:iceprog_includes"],
    deps = [
        ":io_strategy",
        "//xls/codegen:vast",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/tools:verilog_include",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "wrap_io_main",
    srcs = ["wrap_io_main.cc"],
    deps = [
        ":ice40_io_strategy_registry",
        ":io_strategy",
        ":io_strategy_factory",
        ":wrap_io",
        "//xls/codegen:module_signature",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ice40_io_strategy_registry",
    srcs = ["ice40_io_strategy_registry.cc"],
    deps = [
        ":ice40_io_strategy",
        ":io_strategy_factory",
        "//xls/common:module_initializer",
    ],
    alwayslink = True,
)

cc_binary(
    name = "drpc_main",
    srcs = ["drpc_main.cc"],
    deps = [
        ":device_rpc_strategy",
        ":device_rpc_strategy_factory",
        ":ice40_device_rpc_strategy_registry",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ice40_device_rpc_strategy",
    srcs = ["ice40_device_rpc_strategy.cc"],
    hdrs = ["ice40_device_rpc_strategy.h"],
    deps = [
        ":device_rpc_strategy",
        "//xls/common:math_util",
        "//xls/common:strerror",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ice40_device_rpc_strategy_registry",
    srcs = ["ice40_device_rpc_strategy_registry.cc"],
    deps = [
        ":device_rpc_strategy_factory",
        ":ice40_device_rpc_strategy",
        "//xls/common:module_initializer",
    ],
    alwayslink = True,
)

cc_library(
    name = "null_io_strategy",
    srcs = ["null_io_strategy.cc"],
    hdrs = ["null_io_strategy.h"],
    deps = [
        ":io_strategy",
        "//xls/codegen:vast",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "device_rpc_strategy",
    hdrs = ["device_rpc_strategy.h"],
    deps = [
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "device_rpc_strategy_factory",
    srcs = ["device_rpc_strategy_factory.cc"],
    hdrs = ["device_rpc_strategy_factory.h"],
    deps = [
        ":device_rpc_strategy",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "io_strategy",
    hdrs = ["io_strategy.h"],
    deps = [
        "//xls/codegen:vast",
        "//xls/tools:verilog_include",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "io_strategy_factory",
    srcs = ["io_strategy_factory.cc"],
    hdrs = ["io_strategy_factory.h"],
    deps = [
        ":io_strategy",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)
