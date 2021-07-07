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

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ir_interpreter",
    srcs = [
        "block_interpreter.cc",
        "function_interpreter.cc",
        "ir_interpreter.cc",
    ],
    hdrs = [
        "block_interpreter.h",
        "function_interpreter.h",
        "ir_interpreter.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:keyword_args",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
    ],
)

cc_test(
    name = "ir_interpreter_test",
    size = "small",
    srcs = ["ir_interpreter_test.cc"],
    deps = [
        ":ir_evaluator_test_base",
        ":ir_interpreter",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_interpreter",
    srcs = ["proc_interpreter.cc"],
    hdrs = ["proc_interpreter.h"],
    deps = [
        ":channel_queue",
        ":ir_interpreter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/logging:log_lines",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
    ],
)

cc_test(
    name = "proc_interpreter_test",
    srcs = ["proc_interpreter_test.cc"],
    deps = [
        ":proc_interpreter",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "channel_queue_test",
    srcs = ["channel_queue_test.cc"],
    deps = [
        ":channel_queue",
        ":ir_interpreter",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:channel",
        "//xls/ir:channel_cc_proto",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "proc_network_interpreter_test",
    srcs = ["proc_network_interpreter_test.cc"],
    deps = [
        ":proc_network_interpreter",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "block_interpreter_test",
    srcs = ["block_interpreter_test.cc"],
    deps = [
        ":ir_interpreter",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ir_evaluator_test_base",
    testonly = True,
    srcs = ["ir_evaluator_test_base.cc"],
    hdrs = ["ir_evaluator_test_base.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:keyword_args",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "channel_queue",
    srcs = ["channel_queue.cc"],
    hdrs = ["channel_queue.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
    ],
)

cc_library(
    name = "proc_network_interpreter",
    srcs = ["proc_network_interpreter.cc"],
    hdrs = ["proc_network_interpreter.h"],
    deps = [
        ":channel_queue",
        ":proc_interpreter",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/ir",
    ],
)
