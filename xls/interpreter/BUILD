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
        "//xls/codegen:module_signature_cc_proto",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:events",
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
        ":proc_evaluator",
        ":serial_proc_runtime",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
    ],
)

cc_test(
    name = "proc_interpreter_test",
    srcs = ["proc_interpreter_test.cc"],
    deps = [
        ":channel_queue",
        ":proc_evaluator",
        ":proc_evaluator_test_base",
        ":proc_interpreter",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "channel_queue_test",
    srcs = ["channel_queue_test.cc"],
    deps = [
        ":channel_queue",
        ":channel_queue_test_base",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:channel",
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
        "//xls/ir:events",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:keyword_args",
        "//xls/ir:value_helpers",
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
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
    ],
)

cc_library(
    name = "random_value",
    srcs = ["random_value.cc"],
    hdrs = ["random_value.h"],
    deps = [
        ":ir_interpreter",
        "@com_google_absl//absl/status",
        "//xls/ir",
        "//xls/ir:type",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "channel_queue_test_base",
    testonly = True,
    srcs = ["channel_queue_test_base.cc"],
    hdrs = ["channel_queue_test_base.h"],
    deps = [
        ":channel_queue",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:matchers",
        "//xls/ir:channel",
        "//xls/ir:channel_cc_proto",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_evaluator",
    srcs = ["proc_evaluator.cc"],
    hdrs = ["proc_evaluator.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "proc_evaluator_test_base",
    testonly = True,
    srcs = ["proc_evaluator_test_base.cc"],
    hdrs = ["proc_evaluator_test_base.h"],
    deps = [
        ":channel_queue",
        ":proc_evaluator",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_runtime",
    srcs = ["proc_runtime.cc"],
    hdrs = ["proc_runtime.h"],
    deps = [
        ":channel_queue",
        ":proc_evaluator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/jit:jit_channel_queue",
    ],
)

cc_library(
    name = "serial_proc_runtime",
    srcs = ["serial_proc_runtime.cc"],
    hdrs = ["serial_proc_runtime.h"],
    deps = [
        ":channel_queue",
        ":proc_evaluator",
        ":proc_runtime",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/ir",
    ],
)

cc_test(
    name = "serial_proc_runtime_test",
    srcs = ["serial_proc_runtime_test.cc"],
    deps = [
        ":interpreter_proc_runtime",
        ":proc_interpreter",
        ":proc_runtime_test_base",
        ":serial_proc_runtime",
        "//xls/common:xls_gunit_main",
        "//xls/ir",
        "//xls/jit:jit_channel_queue",
        "//xls/jit:jit_proc_runtime",
        "//xls/jit:proc_jit",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "proc_runtime_test_base",
    testonly = True,
    srcs = ["proc_runtime_test_base.cc"],
    hdrs = ["proc_runtime_test_base.h"],
    deps = [
        ":proc_runtime",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "interpreter_proc_runtime",
    srcs = ["interpreter_proc_runtime.cc"],
    hdrs = ["interpreter_proc_runtime.h"],
    deps = [
        ":channel_queue",
        ":proc_evaluator",
        ":proc_interpreter",
        ":serial_proc_runtime",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)
