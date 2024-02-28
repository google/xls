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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_ir_cc_library",
)
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "observer",
    srcs = ["observer.cc"],
    hdrs = ["observer.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_binary(
    name = "aot_compiler",
    srcs = ["aot_compiler.cc"],
    deps = [
        ":function_jit",
        ":llvm_type_converter",
        ":orc_jit",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "aot_compiler_test",
    srcs = ["aot_compiler_test.cc"],
    # The XLS AOT compiler does not currently support cross-compilation.
    deps = [
        ":compound_type_cc",
        ":null_function_cc",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/stdlib:float32_add_cc",
        "//xls/dslx/stdlib:float32_fma_cc",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "aot_runtime",
    srcs = ["aot_runtime.cc"],
    hdrs = ["aot_runtime.h"],
    deps = [
        ":type_layout",
        ":type_layout_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "ir_builder_visitor",
    srcs = ["ir_builder_visitor.cc"],
    hdrs = ["ir_builder_visitor.h"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":llvm_type_converter",
        ":orc_jit",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:elaboration",
        "//xls/ir:events",
        "//xls/ir:format_strings",
        "//xls/ir:op",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "jit_wrapper_generator",
    srcs = ["jit_wrapper_generator.cc"],
    hdrs = ["jit_wrapper_generator.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
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
        "//xls/common:exit_status",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_parser",
    ],
)

alias(
    name = "switchable_function_jit",
    actual = "switchable_function_jit-normal",
    # Uncomment this and comment the above to make the
    # switchable-function-jit use the interpreter instead. This can be used
    # if the jit is non-functional due to some llvm update.
    # actual = "switchable_function_jit-interpreter_fallback",
    visibility = ["//xls:xls_users"],
)

cc_library(
    name = "switchable_function_jit-normal",
    srcs = ["switchable_function_jit.cc"],
    hdrs = ["switchable_function_jit.h"],
    defines = [
        "EMERGENCY_FORCE_INTERPRETER=0",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":function_jit",
        ":observer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "switchable_function_jit-interpreter_fallback",
    srcs = ["switchable_function_jit.cc"],
    hdrs = ["switchable_function_jit.h"],
    defines = [
        "EMERGENCY_FORCE_INTERPRETER=1",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":function_jit",
        ":observer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:value",
    ],
)

cc_test(
    name = "switchable_function_jit_test",
    srcs = ["switchable_function_jit_test.cc"],
    deps = [
        ":switchable_function_jit",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
    ],
)

cc_test(
    name = "switchable_function_jit_interpreter_fallback_test",
    srcs = ["switchable_function_jit_interpreter_fallback_test.cc"],
    deps = [
        ":switchable_function_jit-interpreter_fallback",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "function_jit",
    srcs = ["function_jit.cc"],
    hdrs = ["function_jit.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":function_base_jit",
        ":jit_buffer",
        ":jit_runtime",
        ":observer",
        ":orc_jit",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:keyword_args",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_test(
    name = "function_jit_test",
    srcs = ["function_jit_test.cc"],
    # 2020-01-08 //xls/jit/function_jit_test \
    #               --gtest_list_tests 2>/dev/null | wc -l
    # shows ~140
    shard_count = 50,
    deps = [
        ":function_base_jit",
        ":function_jit",
        ":jit_buffer",
        ":jit_runtime",
        ":orc_jit",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_evaluator_test_base",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:fuzz_type_domain",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_view",
        "@com_google_fuzztest//fuzztest",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "jit_channel_queue",
    srcs = ["jit_channel_queue.cc"],
    hdrs = ["jit_channel_queue.h"],
    deps = [
        ":jit_runtime",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:elaboration",
        "//xls/ir:value",
    ],
)

cc_test(
    name = "jit_channel_queue_test",
    srcs = ["jit_channel_queue_test.cc"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":orc_jit",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:channel_queue_test_base",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:elaboration",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "jit_runtime",
    srcs = ["jit_runtime.cc"],
    hdrs = ["jit_runtime.h"],
    deps = [
        ":llvm_type_converter",
        ":orc_jit",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # build_cleaner: keep
    ],
)

cc_library(
    name = "llvm_type_converter",
    srcs = ["llvm_type_converter.cc"],
    hdrs = ["llvm_type_converter.h"],
    deps = [
        ":type_layout",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:type",
        "//xls/ir:value",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "orc_jit",
    srcs = ["orc_jit.cc"],
    hdrs = ["orc_jit.h"],
    deps = [
        ":observer",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IRPrinter",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:JITLink",  # build_cleaner: keep
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:X86AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "block_jit",
    srcs = ["block_jit.cc"],
    hdrs = ["block_jit.h"],
    deps = [
        ":function_base_jit",
        ":jit_buffer",
        ":jit_runtime",
        ":orc_jit",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:block_evaluator",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:register",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
    ],
)

cc_test(
    name = "block_jit_test",
    srcs = ["block_jit_test.cc"],
    deps = [
        ":block_jit",
        ":jit_runtime",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/interpreter:block_evaluator_test_base",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/ir:value_view",
    ],
)

cc_library(
    name = "proc_jit",
    srcs = ["proc_jit.cc"],
    hdrs = ["proc_jit.h"],
    deps = [
        ":function_base_jit",
        ":ir_builder_visitor",
        ":jit_buffer",
        ":jit_channel_queue",
        ":jit_runtime",
        ":observer",
        ":orc_jit",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:proc_evaluator",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:elaboration",
        "//xls/ir:events",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "jit_buffer",
    srcs = ["jit_buffer.cc"],
    hdrs = ["jit_buffer.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
    ],
)

cc_library(
    name = "function_base_jit",
    srcs = ["function_base_jit.cc"],
    hdrs = ["function_base_jit.h"],
    deps = [
        ":ir_builder_visitor",
        ":jit_buffer",
        ":jit_channel_queue",
        ":jit_runtime",
        ":llvm_type_converter",
        ":orc_jit",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:type",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "jit_proc_runtime",
    srcs = ["jit_proc_runtime.cc"],
    hdrs = ["jit_proc_runtime.h"],
    deps = [
        ":jit_channel_queue",
        ":proc_jit",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:proc_evaluator",
        "//xls/interpreter:proc_interpreter",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:elaboration",
        "//xls/ir:value",
    ],
)

cc_binary(
    name = "value_to_native_layout_benchmark",
    srcs = ["value_to_native_layout_benchmark.cc"],
    deps = [
        ":llvm_type_converter",
        ":orc_jit",
        ":type_layout",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "@com_google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "jit_channel_queue_benchmark",
    srcs = ["jit_channel_queue_benchmark.cc"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        "@com_google_absl//absl/log:check",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:elaboration",
        "@com_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "type_layout",
    srcs = ["type_layout.cc"],
    hdrs = ["type_layout.h"],
    deps = [
        ":type_layout_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
    ],
)

cc_test(
    name = "type_layout_test",
    srcs = ["type_layout_test.cc"],
    deps = [
        ":llvm_type_converter",
        ":orc_jit",
        ":type_layout",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:bits_util",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/data_structures:leaf_type_tree",
        "//xls/interpreter:random_value",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:number_parser",
        "//xls/ir:type",
    ],
)

cc_test(
    name = "proc_jit_test",
    srcs = ["proc_jit_test.cc"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":orc_jit",
        ":proc_jit",
        "@com_google_absl//absl/log:check",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:proc_evaluator",
        "//xls/interpreter:proc_evaluator_test_base",
        "//xls/ir",
    ],
)

build_test(
    name = "metadata_proto_libraries_build",
    targets = [
        ":jit_channel_queue_benchmark",
        ":value_to_native_layout_benchmark",
    ],
)

proto_library(
    name = "type_layout_proto",
    srcs = ["type_layout.proto"],
)

cc_proto_library(
    name = "type_layout_cc_proto",
    deps = [":type_layout_proto"],
)

xls_dslx_library(
    name = "null_function_dslx",
    srcs = ["null_function.x"],
)

xls_dslx_opt_ir(
    name = "null_function",
    dslx_top = "null_function",
    library = ":null_function_dslx",
)

xls_ir_cc_library(
    name = "null_function_cc",
    src = ":null_function.ir",
    namespaces = "xls,foo,bar",
    top = "null_function",
)

xls_dslx_library(
    name = "compound_type_dslx",
    srcs = ["compound_type.x"],
)

xls_dslx_opt_ir(
    name = "compound_type",
    dslx_top = "fun_test_function",
    library = ":compound_type_dslx",
)

xls_ir_cc_library(
    name = "compound_type_cc",
    src = ":compound_type.ir",
    namespaces = "xls",
    top = "fun_test_function",
)
