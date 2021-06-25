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

# Build rules for the IR JIT - converts XLS IR into native host code.

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "function_builder_visitor",
    srcs = ["function_builder_visitor.cc"],
    hdrs = ["function_builder_visitor.h"],
    deps = [
        ":jit_runtime",
        ":llvm_type_converter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:vast",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "jit_wrapper_generator",
    srcs = ["jit_wrapper_generator.cc"],
    hdrs = ["jit_wrapper_generator.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common/status:ret_check",
        "//xls/ir",
    ],
)

cc_binary(
    name = "jit_wrapper_generator_main",
    srcs = ["jit_wrapper_generator_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":jit_wrapper_generator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common:case_converters",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
    ],
)

cc_test(
    name = "jit_wrapper_generator_test",
    srcs = ["jit_wrapper_generator_test.cc"],
    deps = [
        ":jit_wrapper_generator",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ir_jit",
    srcs = ["ir_jit.cc"],
    hdrs = ["ir_jit.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":function_builder_visitor",
        ":jit_channel_queue",
        ":jit_runtime",
        ":llvm_type_converter",
        ":proc_builder_visitor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:vast",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:format_preference",
        "//xls/ir:keyword_args",
        "//xls/ir:random_value",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:JITLink",  # build_cleaner: keep
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # build_cleaner: keep
    ],
)

cc_test(
    name = "ir_jit_test",
    srcs = ["ir_jit_test.cc"],
    # 2020-01-08 //xls/jit/ir_jit_test \
    #               --gtest_list_tests 2>/dev/null | wc -l
    # shows ~140
    shard_count = 50,
    deps = [
        ":ir_jit",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:ir_evaluator_test_base",
        "//xls/ir:function_builder",
        "//xls/ir:random_value",
        "@com_github_google_re2//:re2",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "jit_channel_queue",
    srcs = ["jit_channel_queue.cc"],
    hdrs = ["jit_channel_queue.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
    ],
)

cc_library(
    name = "jit_runtime",
    srcs = ["jit_runtime.cc"],
    hdrs = ["jit_runtime.h"],
    deps = [
        ":llvm_type_converter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/interpreter:channel_queue",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # build_cleaner: keep
    ],
)

cc_library(
    name = "llvm_type_converter",
    srcs = ["llvm_type_converter.cc"],
    hdrs = ["llvm_type_converter.h"],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "proc_builder_visitor",
    srcs = ["proc_builder_visitor.cc"],
    hdrs = ["proc_builder_visitor.h"],
    deps = [
        ":function_builder_visitor",
        ":jit_channel_queue",
        ":llvm_type_converter",
        "//xls/ir",
        "@llvm-project//llvm:Core",
    ],
)

cc_test(
    name = "proc_builder_visitor_test",
    srcs = ["proc_builder_visitor_test.cc"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":llvm_type_converter",
        ":proc_builder_visitor",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:subprocess",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MCJIT",  # build_cleaner: keep
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Target",  # build_cleaner: keep
    ],
)

cc_library(
    name = "serial_proc_runtime",
    srcs = ["serial_proc_runtime.cc"],
    hdrs = ["serial_proc_runtime.h"],
    deps = [
        ":function_builder_visitor",
        ":ir_jit",
        ":jit_channel_queue",
        ":proc_builder_visitor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:thread",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_test(
    name = "serial_proc_runtime_test",
    srcs = ["serial_proc_runtime_test.cc"],
    deps = [
        ":jit_channel_queue",
        ":serial_proc_runtime",
        "//xls/common:thread",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@com_google_googletest//:gtest",
    ],
)
