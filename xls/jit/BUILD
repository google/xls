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

# cc_proto_library is used in this file
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

# pytype binary and test
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")

# Load proto_library
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_ir_cc_library",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load(
    "//xls/build_rules:xls_internal_aot_rules.bzl",
    "xls_aot_generate",
)
load(
    "//xls/build_rules:xls_internal_build_defs.bzl",
    "XLS_IS_MSAN_BUILD",
)
load(":aot_test_helper.bzl", "aot_protobuf")

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
    name = "observer",
    srcs = ["observer.cc"],
    hdrs = ["observer.h"],
    deps = [
        ":jit_runtime",
        "//xls/interpreter:observer",
        "//xls/ir",
        "//xls/ir:value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "jit_callbacks",
    srcs = ["jit_callbacks.cc"],
    hdrs = ["jit_callbacks.h"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":observer",
        "//xls/ir:events",
        "//xls/ir:format_preference",
        "//xls/ir:proc_elaboration",
        "//xls/ir:type",
        "//xls/ir:type_manager",
        "//xls/ir:value",
        "//xls/ir:xls_type_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "aot_compiler_main",
    srcs = ["aot_compiler_main.cc"],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":block_jit",
        ":function_base_jit",
        ":function_jit",
        ":jit_proc_runtime",
        ":llvm_compiler",
        ":llvm_type_converter",
        ":observer",
        ":type_layout_cc_proto",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dev_tools:extract_interface",
        "//xls/ir",
        "//xls/ir:block_elaboration",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

py_binary(
    name = "aot_basic_function_entrypoint_main",
    srcs = ["aot_basic_function_entrypoint_main.py"],
    data = [
        ":aot_basic_function_cc.tmpl",
        ":aot_basic_function_h.tmpl",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":aot_entrypoint_py_pb2",
        ":type_layout_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)

cc_test(
    name = "aot_basic_function_entrypoint_test",
    srcs = ["aot_basic_function_entrypoint_test.cc"],
    # The XLS AOT compiler does not currently support cross-compilation.
    deps = [
        ":compound_type_cc",
        ":multi_function_one_cc",
        ":multi_function_two_cc",
        ":null_function_cc",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/stdlib:float32_add_cc",
        "//xls/dslx/stdlib:float32_fma_cc",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "aot_runtime",
    srcs = ["aot_runtime.cc"],
    hdrs = ["aot_runtime.h"],
    deps = [
        ":type_layout",
        ":type_layout_cc_proto",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:value",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "ir_builder_visitor",
    srcs = ["ir_builder_visitor.cc"],
    hdrs = ["ir_builder_visitor.h"],
    deps = [
        ":jit_callbacks",
        ":llvm_compiler",
        ":llvm_type_converter",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
        "//xls/ir:format_strings",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "function_base_jit_wrapper",
    hdrs = ["function_base_jit_wrapper.h"],
    # Allow jit-wrapper users to see this.
    visibility = ["//xls:xls_users"],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":function_jit",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_view",
        "//xls/ir:xls_ir_interface_cc_proto",
        "//xls/ir:xls_type_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "block_base_jit_wrapper",
    srcs = ["block_base_jit_wrapper.cc"],
    hdrs = ["block_base_jit_wrapper.h"],
    # Allow jit-wrapper users to see this.
    visibility = ["//xls:xls_users"],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":block_jit",
        ":function_base_jit",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:evaluator_options",
        "//xls/ir:events",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "proc_base_jit_wrapper",
    hdrs = ["proc_base_jit_wrapper.h"],
    # Allow jit-wrapper users to see this.
    visibility = ["//xls:xls_users"],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":jit_channel_queue",
        ":jit_proc_runtime",
        ":jit_runtime",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:evaluator_options",
        "//xls/interpreter:proc_runtime",
        "//xls/ir",
        "//xls/ir:state_element",
        "//xls/ir:value",
        "//xls/ir:xls_ir_interface_cc_proto",
        "//xls/public:ir_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

py_binary(
    name = "jit_wrapper_generator_main",
    srcs = ["jit_wrapper_generator_main.py"],
    data = [
        ":jit_block_wrapper_cc.tmpl",
        ":jit_block_wrapper_h.tmpl",
        ":jit_function_wrapper_cc.tmpl",
        ":jit_function_wrapper_h.tmpl",
        ":jit_proc_wrapper_cc.tmpl",
        ":jit_proc_wrapper_h.tmpl",
        "//xls/dev_tools:extract_interface_main",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":aot_entrypoint_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "//xls/ir:xls_ir_interface_py_pb2",
        "//xls/ir:xls_type_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

cc_test(
    name = "jit_wrapper_test",
    srcs = ["jit_wrapper_test.cc"],
    deps = [
        ":compound_type_jit_wrapper",
        ":multi_func_block_wrapper",
        ":multi_func_with_trace_block_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/stdlib:float32_mul_jit_wrapper",
        "//xls/dslx/stdlib/tests:float32_upcast_jit_wrapper",
        "//xls/examples/dslx_module:some_caps_jit_wrapper",
        "//xls/examples/dslx_module:some_caps_opt_jit_wrapper",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_builder",
        "//xls/ir:value_view",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "llvm_compiler",
    srcs = ["llvm_compiler.cc"],
    hdrs = ["llvm_compiler.h"],
    deps = [
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "jit_emulated_tls",
    srcs = ["jit_emulated_tls.cc"],
    hdrs = ["jit_emulated_tls.h"],
    deps = [
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "switchable_function_jit",
    srcs = ["switchable_function_jit.cc"],
    hdrs = ["switchable_function_jit.h"],
    defines = [
        "EMERGENCY_FORCE_INTERPRETER=0",
    ],
    visibility = ["//xls:xls_users"],
    deps = [
        ":function_jit",
        ":observer",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "switchable_function_jit_test",
    srcs = ["switchable_function_jit_test.cc"],
    deps = [
        ":switchable_function_jit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "function_jit",
    srcs = ["function_jit.cc"],
    hdrs = ["function_jit.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":aot_compiler",
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":jit_buffer",
        ":jit_callbacks",
        ":jit_runtime",
        ":observer",
        ":orc_jit",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:keyword_args",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_test(
    name = "function_jit_test",
    timeout = "long",
    srcs = ["function_jit_test.cc"],
    # 2024-07-30 <//xls/jit:function_jit_test binary>
    #               --gunit_list_tests 2>/dev/null | wc -l
    # shows ~200
    shard_count = 50,
    deps = [
        ":function_base_jit",
        ":function_jit",
        ":jit_buffer",
        ":jit_runtime",
        ":llvm_compiler",
        ":observer",
        ":orc_jit",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_evaluator_test_base",
        "//xls/interpreter:observer",
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
        "//xls/ir:xls_type_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
    ],
)

# Separate test since for the aot stuff to build at all function-jit needs to be *mostly* working.
cc_test(
    name = "function_jit_aot_test",
    srcs = [
        "function_jit_aot_test.cc",
    ],
    data = [
        ":multi_function_with_trace.ir",
        ":multi_function_with_trace_aot_pb",
    ],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":function_jit",
        ":jit_callbacks",
        ":jit_runtime",
        ":multi_function_with_trace_aot",  # build_cleaner: keep
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_view",
        "//xls/public:ir_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "proc_jit_aot_test",
    srcs = [
        "proc_jit_aot_test.cc",
    ],
    data = [
        ":multi_proc.ir",
        ":multi_proc_aot",
        ":some_caps_no_idents.ir",
        ":specialized_caps_aot",
    ],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":jit_callbacks",
        ":jit_channel_queue",
        ":jit_proc_runtime",
        ":jit_runtime",
        ":multi_proc_aot",  # build_cleaner: keep
        ":specialized_caps_aot",  # build_cleaner: keep
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dev_tools:extract_interface",
        "//xls/interpreter:channel_queue",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:type_manager",
        "//xls/ir:value",
        "//xls/ir:value_builder",
        "//xls/ir:value_utils",
        "//xls/ir:xls_ir_interface_cc_proto",
        "//xls/public:ir_parser",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "jit_channel_queue",
    srcs = ["jit_channel_queue.cc"],
    hdrs = ["jit_channel_queue.h"],
    deps = [
        ":jit_runtime",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:proc_elaboration",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "jit_channel_queue_test",
    srcs = ["jit_channel_queue_test.cc"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":orc_jit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/interpreter:channel_queue_test_base",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:proc_elaboration",
        "//xls/ir:value",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "jit_runtime",
    srcs = ["jit_runtime.cc"],
    hdrs = ["jit_runtime.h"],
    deps = [
        ":llvm_type_converter",
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "//xls/ir:bits",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
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
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "aot_compiler",
    srcs = ["aot_compiler.cc"],
    hdrs = ["aot_compiler.h"],
    visibility = [
        "//xls:xls_internal",
    ],
    deps = [
        ":jit_emulated_tls",
        ":llvm_compiler",
        ":observer",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:X86AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:X86CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "jit_clang_builtins",
    srcs = ["jit_clang_builtins.cc"],
    hdrs = ["jit_clang_builtins.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "orc_jit",
    srcs = ["orc_jit.cc"],
    hdrs = ["orc_jit.h"],
    deps = [
        ":jit_clang_builtins",
        ":jit_emulated_tls",
        ":llvm_compiler",
        ":observer",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:AArch64AsmParser",  # build_cleaner: keep
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IRPrinter",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
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
        ":aot_compiler",
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":jit_buffer",
        ":jit_callbacks",
        ":jit_runtime",
        ":llvm_compiler",
        ":observer",
        ":orc_jit",
        "//xls/codegen:block_inlining_pass",
        "//xls/codegen:codegen_options",
        "//xls/codegen:codegen_pass",
        "//xls/codegen:materialize_fifos_pass",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:block_evaluator",
        "//xls/interpreter:observer",
        "//xls/ir",
        "//xls/ir:block_elaboration",
        "//xls/ir:clone_package",
        "//xls/ir:elaboration",
        "//xls/ir:events",
        "//xls/ir:register",
        "//xls/ir:type",
        "//xls/ir:type_manager",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_test(
    name = "block_jit_test",
    srcs = ["block_jit_test.cc"],
    deps = [
        ":block_jit",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/interpreter:block_evaluator_test_base",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:foreign_function_data_cc_proto",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:register",
        "//xls/ir:value",
        "//xls/ir:value_view",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "proc_jit",
    srcs = ["proc_jit.cc"],
    hdrs = ["proc_jit.h"],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":jit_buffer",
        ":jit_callbacks",
        ":jit_channel_queue",
        ":jit_runtime",
        ":llvm_compiler",
        ":observer",
        ":orc_jit",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:observer",
        "//xls/interpreter:proc_evaluator",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/ir:proc_elaboration",
        "//xls/ir:state_element",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "jit_buffer",
    srcs = ["jit_buffer.cc"],
    hdrs = ["jit_buffer.h"],
    deps = [
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function_base_jit",
    srcs = ["function_base_jit.cc"],
    hdrs = ["function_base_jit.h"],
    deps = [
        ":aot_entrypoint_cc_proto",
        ":ir_builder_visitor",
        ":jit_buffer",
        ":jit_callbacks",
        ":jit_runtime",
        ":llvm_compiler",
        ":llvm_type_converter",
        ":orc_jit",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:state_element",
        "//xls/ir:type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "jit_proc_runtime",
    srcs = ["jit_proc_runtime.cc"],
    hdrs = ["jit_proc_runtime.h"],
    deps = [
        ":aot_compiler",
        ":aot_entrypoint_cc_proto",
        ":function_base_jit",
        ":jit_channel_queue",
        ":jit_runtime",
        ":llvm_compiler",
        ":observer",
        ":proc_jit",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:evaluator_options",
        "//xls/interpreter:proc_evaluator",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:proc_elaboration",
        "//xls/ir:value",
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_binary(
    name = "value_to_native_layout_benchmark",
    testonly = True,
    srcs = ["value_to_native_layout_benchmark.cc"],
    deps = [
        ":llvm_type_converter",
        ":orc_jit",
        ":type_layout",
        "//xls/common:benchmark_support",
        "//xls/common:init_xls",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "@google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "jit_channel_queue_benchmark",
    testonly = True,
    srcs = ["jit_channel_queue_benchmark.cc"],
    deps = [
        ":jit_channel_queue",
        ":jit_runtime",
        ":orc_jit",
        "//xls/common:benchmark_support",
        "//xls/common:init_xls",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:proc_elaboration",
        "@com_google_absl//absl/log:check",
        "@google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "type_layout",
    srcs = ["type_layout.cc"],
    hdrs = ["type_layout.h"],
    deps = [
        ":type_layout_cc_proto",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "type_layout_test",
    srcs = ["type_layout_test.cc"],
    deps = [
        ":llvm_type_converter",
        ":orc_jit",
        ":type_layout",
        "//xls/common:bits_util",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/data_structures:leaf_type_tree",
        "//xls/interpreter:random_value",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:number_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:proc_evaluator",
        "//xls/interpreter:proc_evaluator_test_base",
        "//xls/ir",
        "@com_google_absl//absl/log:check",
        "@googletest//:gtest",
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
    name = "aot_entrypoint_proto",
    srcs = ["aot_entrypoint.proto"],
    deps = [
        ":type_layout_proto",
        "//xls/ir:xls_ir_interface_proto",
        "//xls/ir:xls_type_proto",
    ],
)

cc_proto_library(
    name = "aot_entrypoint_cc_proto",
    deps = [":aot_entrypoint_proto"],
)

py_proto_library(
    name = "aot_entrypoint_py_pb2",
    deps = [":aot_entrypoint_proto"],
)

proto_library(
    name = "type_layout_proto",
    srcs = ["type_layout.proto"],
)

cc_proto_library(
    name = "type_layout_cc_proto",
    deps = [":type_layout_proto"],
)

py_proto_library(
    name = "type_layout_py_pb2",
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
    name = "multi_function_dslx",
    srcs = ["multi_func.x"],
)

xls_dslx_ir(
    name = "multi_function",
    dslx_top = "multi_function_one",
    library = ":multi_function_dslx",
)

xls_ir_opt_ir(
    name = "multi_func_opt_ir",
    src = ":multi_function",
)

xls_ir_verilog(
    name = "multi_func_sv",
    src = ":multi_func_opt_ir",
    codegen_args = {
        "module_name": "multi_func_top",
        "delay_model": "unit",
        "generator": "combinational",
        "reset": "rst",
        "reset_data_path": "true",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "multi_proc": "true",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
    },
    verilog_file = "multi_func.sv",
)

cc_xls_ir_jit_wrapper(
    name = "multi_func_block_wrapper",
    src = ":multi_func.block.ir",
    jit_wrapper_args = {
        "class_name": "MultiFuncBlockJit",
        "namespace": "something::cool",
    },
    top = "multi_func_top",
    wrapper_type = "BLOCK",
)

xls_aot_generate(
    name = "multi_function_aot",
    testonly = True,
    src = ":multi_function",
    top = "multi_function_one",
    with_msan = XLS_IS_MSAN_BUILD,
)

# This is needed since xls_aot_generate isn't really meant to be used directly
# so doesn't register the protobuf as a pre-declared output file, complicating its use in a genrule.
aot_protobuf(
    name = "multi_function_aot_pb",
    testonly = True,
    aot = ":multi_function_aot",
)

xls_dslx_library(
    name = "multi_function_with_trace_dslx",
    srcs = ["multi_func_with_trace.x"],
)

xls_dslx_ir(
    name = "multi_function_with_trace",
    dslx_top = "multi_function_one",
    library = ":multi_function_with_trace_dslx",
)

xls_ir_opt_ir(
    name = "multi_function_with_trace_opt_ir",
    src = ":multi_function_with_trace",
)

# Codegen settings produce registers for this block. We rely on this in a jit_wrapper_test.
xls_ir_verilog(
    name = "multi_function_with_trace_sv",
    src = ":multi_function_with_trace_opt_ir",
    codegen_args = {
        "module_name": "multi_func_with_trace_top",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "1",
    },
    verilog_file = "multi_func_with_trace.sv",
)

cc_xls_ir_jit_wrapper(
    name = "multi_func_with_trace_block_wrapper",
    src = ":multi_func_with_trace.block.ir",
    jit_wrapper_args = {
        "class_name": "MultiFuncWithTraceBlockJit",
        "namespace": "something::cool",
    },
    top = "multi_func_with_trace_top",
    wrapper_type = "BLOCK",
)

xls_ir_cc_library(
    name = "multi_function_one_cc",
    src = ":multi_function.ir",
    namespaces = "xls,foo,bar",
    top = "multi_function_one",
)

xls_ir_cc_library(
    name = "multi_function_two_cc",
    src = ":multi_function.ir",
    namespaces = "xls,foo,bar",
    top = "multi_function_two",
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

cc_xls_ir_jit_wrapper(
    name = "compound_type_jit_wrapper",
    src = ":compound_type.ir",
    jit_wrapper_args = {
        "class_name": "CompoundJitWrapper",
        "namespace": "something::cool",
    },
)

xls_aot_generate(
    name = "multi_function_with_trace_aot",
    src = ":multi_function_with_trace.ir",
    top = "multi_function_one",
    with_msan = XLS_IS_MSAN_BUILD,
)

aot_protobuf(
    name = "multi_function_with_trace_aot_pb",
    testonly = True,
    aot = ":multi_function_with_trace_aot",
)

# Procs get the file name embedded in their identifier when they have generics.
# This means that writing a test which also works in OSS would be difficult. To
# avoid this problem we just remove all identifiers. Normal users should never
# need to do this since the generated wrappers handle all of this for you.
genrule(
    name = "some_caps_no_idents",
    srcs = ["//xls/examples/dslx_module:manual_chan_caps_streaming_configured_opt_ir.opt.ir"],
    outs = ["some_caps_no_idents.ir"],
    cmd = """
        $(location //xls/dev_tools:remove_identifiers_main) $< > $@
        """,
    tools = ["//xls/dev_tools:remove_identifiers_main"],
)

xls_aot_generate(
    name = "specialized_caps_aot",
    src = ":some_caps_no_idents",
    with_msan = XLS_IS_MSAN_BUILD,
)

xls_dslx_library(
    name = "multi_proc_dslx",
    srcs = ["multi_proc.x"],
)

xls_dslx_ir(
    name = "multi_proc",
    dslx_top = "proc_ten",
    library = ":multi_proc_dslx",
)

xls_aot_generate(
    name = "multi_proc_aot",
    src = ":multi_proc",
    with_msan = XLS_IS_MSAN_BUILD,
)

cc_binary(
    name = "type_layout_main",
    srcs = ["type_layout_main.cc"],
    deps = [
        ":type_layout",
        ":type_layout_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_test(
    name = "type_layout_main_test",
    srcs = ["type_layout_main_test.py"],
    data = [":type_layout_main"],
    deps = [
        ":type_layout_py_pb2",
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_binary(
    name = "aot_main_wrapper_main",
    srcs = ["aot_main_wrapper_main.py"],
    data = [
        ":aot_main_cc.tmpl",
        ":type_layout_main",
    ],
    visibility = ["//xls:xls_internal"],
    deps = [
        ":aot_entrypoint_py_pb2",
        ":type_layout_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

genrule(
    name = "generate_aot_main_wrapper",
    testonly = True,
    srcs = [":multi_function_aot_pb"],
    outs = ["generate_aot_main_wrapper.cc"],
    cmd = """
    $(location :aot_main_wrapper_main) -input 'bits[8]:3' -result 'bits[8]:15' -output $@ -package $(location :multi_function_aot_pb)
    """,
    tools = [":aot_main_wrapper_main"],
)

# Just makes sure that we can execute the wrapper.
cc_test(
    name = "aot_main_wrapper_main_test",
    srcs = [":generate_aot_main_wrapper.cc"],
    deps = [":multi_function_aot"],
)

bzl_library(
    name = "aot_test_helper_bzl",
    srcs = ["aot_test_helper.bzl"],
    visibility = ["//visibility:private"],
    deps = ["//xls/build_rules:xls_providers_bzl"],
)
