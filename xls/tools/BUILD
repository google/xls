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

load("@xls_pip_deps//:requirements.bzl", "requirement")

# pytype binary and test
load("@rules_python//python:proto.bzl", "py_proto_library")
# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "package_bazel_build",
    srcs = ["package_bazel_build.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
    ],
)

sh_test(
    name = "package_bazel_build_test",
    srcs = ["package_bazel_build_test.sh"],
    data = glob(["package_bazel_build_testdata/**"]) + [
        ":package_bazel_build",
    ],
)

py_binary(
    name = "generate_vvp_runner",
    srcs = ["generate_vvp_runner.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl:app",
    ],
)

cc_library(
    name = "booleanifier",
    srcs = ["booleanifier.cc"],
    hdrs = ["booleanifier.h"],
    deps = [
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:abstract_node_evaluator",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "booleanifier_test",
    srcs = ["booleanifier_test.cc"],
    data = ["//xls/examples:crc32.opt.ir"],
    # 2021-01-29: //xls/tools/booleanifier_test --gtest_list_tests
    # shows 4
    shard_count = 4,
    deps = [
        ":booleanifier",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir:ir_test_base",
        "//xls/jit:function_jit",
    ],
)

cc_binary(
    name = "booleanify_main",
    srcs = ["booleanify_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":booleanifier",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "verilog_include",
    hdrs = ["verilog_include.h"],
)

cc_library(
    name = "io_strategy",
    hdrs = ["io_strategy.h"],
    deps = [
        ":verilog_include",
        "//xls/codegen:vast",
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

cc_library(
    name = "remove_identifiers",
    srcs = ["remove_identifiers.cc"],
    hdrs = ["remove_identifiers.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "remove_identifiers_test",
    srcs = ["remove_identifiers_test.cc"],
    deps = [
        ":remove_identifiers",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:register",
        "//xls/ir:source_location",
        "//xls/ir:value",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_binary(
    name = "remove_identifiers_main",
    srcs = ["remove_identifiers_main.cc"],
    deps = [
        ":remove_identifiers",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "ir_stats_main",
    srcs = ["ir_stats_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_binary(
    name = "check_ir_equivalence_main",
    srcs = ["check_ir_equivalence_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes:dce_pass",
        "//xls/passes:inlining_pass",
        "//xls/passes:map_inlining_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/passes:unroll_pass",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@z3//:api",
    ],
)

filegroup(
    name = "check_ir_equivalence_sh",
    srcs = ["check_ir_equivalence.sh"],
    visibility = ["//xls:xls_users"],
)

cc_binary(
    name = "extract_stage_main",
    srcs = ["extract_stage_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "//xls/scheduling:extract_stage",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_binary(
    name = "lec_main",
    srcs = ["lec_main.cc"],
    data = [
        "//xls/dslx/ir_convert:ir_converter_main",
    ],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:function_extractor",
        "//xls/netlist:lib_parser",
        "//xls/netlist:netlist_cc_proto",
        "//xls/netlist:netlist_parser",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/solvers:z3_lec",
        "//xls/solvers:z3_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@z3//:api",
    ],
)

cc_binary(
    name = "netlist_interpreter_main",
    srcs = ["netlist_interpreter_main.cc"],
    deps = [
        "//xls/codegen:flattening",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/netlist:cell_library",
        "//xls/netlist:function_extractor",
        "//xls/netlist:interpreter",
        "//xls/netlist:lib_parser",
        "//xls/netlist:netlist_cc_proto",
        "//xls/netlist:netlist_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

netlist_interpreter_main_test_data = [
    "testdata/sqrt.v",
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

cc_library(
    name = "proto_to_dslx",
    srcs = ["proto_to_dslx.cc"],
    hdrs = ["proto_to_dslx.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        # protobuf compiler import
        "//xls/common:proto_adaptor_utils",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "proto_to_dslx_main",
    srcs = ["proto_to_dslx_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":proto_to_dslx",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/dslx/frontend:module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "proto_to_dslx_test",
    srcs = ["proto_to_dslx_test.cc"],
    deps = [
        ":proto_to_dslx",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:module",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "smtlib_emitter_main",
    srcs = ["smtlib_emitter_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@z3//:api",
    ],
)

cc_library(
    name = "testbench",
    hdrs = ["testbench.h"],
    deps = [
        ":testbench_thread",
        "//xls/common/file:filesystem",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "testbench_builder",
    hdrs = ["testbench_builder.h"],
    deps = [
        ":testbench",
        ":testbench_builder_utils",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "testbench_builder_utils",
    hdrs = ["testbench_builder_utils.h"],
    deps = [
        "//xls/common/logging",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "testbench_thread",
    hdrs = ["testbench_thread.h"],
    deps = [
        "//xls/common:thread",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/jit:function_jit",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
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
        "//xls/ir:type",
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
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir:source_location",
        "//xls/simulation:module_testbench",
        "//xls/simulation:module_testbench_thread",
        "//xls/simulation:verilog_test_base",
    ],
)

cc_library(
    name = "ice40_io_strategy",
    srcs = ["ice40_io_strategy.cc"],
    hdrs = ["ice40_io_strategy.h"],
    data = ["//xls/uncore_rtl/ice40:iceprog_includes"],
    deps = [
        ":wrap_io",
        "//xls/codegen:vast",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
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
        "//xls/common/logging",
        "@com_google_absl//absl/flags:flag",
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
        "//xls/common/logging",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "parse_ir",
    srcs = ["parse_ir.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "@com_google_absl//absl/status",
    ],
)

sh_test(
    name = "parse_ir_test",
    srcs = ["parse_ir_test.sh"],
    data = [
        "testdata/add_folding_overlarge.ir",
        ":parse_ir",
    ],
)

cc_binary(
    name = "eval_dslx_main",
    srcs = ["eval_dslx_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "eval_ir_main",
    srcs = ["eval_ir_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:import_data",
        "//xls/dslx:mangle",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_kind",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:function_jit",
        "//xls/jit:jit_buffer",
        "//xls/jit:llvm_type_converter",
        "//xls/jit:observer",
        "//xls/jit:orc_jit",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Interpreter",  # buildcleaner:keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "eval_utils",
    srcs = ["eval_utils.cc"],
    hdrs = ["eval_utils.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":proc_channel_values_cc_proto",
        "//xls/common:indent",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/ir:xls_value_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "eval_utils_test",
    srcs = ["eval_utils_test.cc"],
    deps = [
        ":eval_utils",
        ":proc_channel_values_cc_proto",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "eval_proc_main",
    srcs = ["eval_proc_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":eval_utils",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:block_evaluator",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:register",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:block_jit",
        "//xls/jit:jit_proc_runtime",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
    name = "ice40_device_rpc_strategy",
    srcs = ["ice40_device_rpc_strategy.cc"],
    hdrs = ["ice40_device_rpc_strategy.h"],
    deps = [
        ":device_rpc_strategy",
        "//xls/common:math_util",
        "//xls/common:strerror",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
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
    name = "opt",
    srcs = ["opt.cc"],
    hdrs = ["opt.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/passes:verifier_checker",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "opt_main",
    srcs = ["opt_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":opt",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ram_rewrite_cc_proto",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

py_test(
    name = "opt_main_test",
    srcs = ["opt_main_test.py"],
    data = [":opt_main"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

cc_binary(
    name = "ir_minimizer_main",
    srcs = ["ir_minimizer_main.cc"],
    data = [
        ":opt_main",
    ],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:binary_search",
        "//xls/data_structures:inline_bitmap",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:function_jit",
        "//xls/passes:arith_simplification_pass",
        "//xls/passes:array_simplification_pass",
        "//xls/passes:bit_slice_simplification_pass",
        "//xls/passes:concat_simplification_pass",
        "//xls/passes:constant_folding_pass",
        "//xls/passes:cse_pass",
        "//xls/passes:dce_pass",
        "//xls/passes:dfe_pass",
        "//xls/passes:inlining_pass",
        "//xls/passes:map_inlining_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/passes:proc_state_flattening_pass",
        "//xls/passes:proc_state_optimization_pass",
        "//xls/passes:tuple_simplification_pass",
        "//xls/passes:unroll_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

py_test(
    name = "mul_folding_overlarge",
    srcs = ["run_opt_main.py"],
    args = ["xls/tools/testdata/mul_folding_overlarge.ir"],
    data = [":opt_main"] + glob(["testdata/*.ir"]),
    main = "run_opt_main.py",
    python_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "sub_folding_overlarge",
    srcs = ["run_opt_main.py"],
    args = ["xls/tools/testdata/sub_folding_overlarge.ir"],
    data = [":opt_main"] + glob(["testdata/*.ir"]),
    main = "run_opt_main.py",
    python_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "sub_add_folding_overlarge",
    srcs = ["run_opt_main.py"],
    args = ["xls/tools/testdata/sub_add_folding_overlarge.ir"],
    data = [":opt_main"] + glob(["testdata/*.ir"]),
    main = "run_opt_main.py",
    python_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "shll_folding_overlarge",
    srcs = ["run_opt_main.py"],
    args = ["xls/tools/testdata/shll_folding_overlarge.ir"],
    data = [":opt_main"] + glob(["testdata/*.ir"]),
    main = "run_opt_main.py",
    python_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "add_folding_overlarge",
    srcs = ["run_opt_main.py"],
    args = ["xls/tools/testdata/add_folding_overlarge.ir"],
    data = [":opt_main"] + glob(["testdata/*.ir"]),
    main = "run_opt_main.py",
    python_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

proto_library(
    name = "proc_channel_values_proto",
    srcs = ["proc_channel_values.proto"],
    visibility = ["//xls:xls_users"],
    deps = ["//xls/ir:xls_value_proto"],
)

cc_proto_library(
    name = "proc_channel_values_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":proc_channel_values_proto"],
)

proto_library(
    name = "scheduling_options_flags_proto",
    srcs = ["scheduling_options_flags.proto"],
    deps = ["@com_google_googleapis//google/api:field_behavior_proto"],
)

py_proto_library(
    name = "scheduling_options_flags_py_pb2",
    deps = [":scheduling_options_flags_proto"],
)

cc_proto_library(
    name = "scheduling_options_flags_cc_proto",
    deps = [":scheduling_options_flags_proto"],
)

cc_library(
    name = "scheduling_options_flags",
    srcs = ["scheduling_options_flags.cc"],
    hdrs = ["scheduling_options_flags.h"],
    deps = [
        ":scheduling_options_flags_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

proto_library(
    name = "codegen_flags_proto",
    srcs = ["codegen_flags.proto"],
)

py_proto_library(
    name = "codegen_flags_py_pb2",
    deps = [":codegen_flags_proto"],
)

cc_proto_library(
    name = "codegen_flags_cc_proto",
    deps = [":codegen_flags_proto"],
)

cc_library(
    name = "codegen_flags",
    srcs = ["codegen_flags.cc"],
    hdrs = ["codegen_flags.h"],
    deps = [
        ":codegen_flags_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "codegen",
    srcs = ["codegen.cc"],
    hdrs = ["codegen.h"],
    deps = [
        ":codegen_flags_cc_proto",
        ":scheduling_options_flags",
        ":scheduling_options_flags_cc_proto",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen:op_override_impls",
        "//xls/codegen:pipeline_generator",
        "//xls/codegen:ram_configuration",
        "//xls/common:stopwatch",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:ffi_delay_estimator",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "//xls/scheduling:scheduling_pass",
        "//xls/scheduling:scheduling_pass_pipeline",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "codegen_main",
    srcs = ["codegen_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":codegen",
        ":codegen_flags",
        ":codegen_flags_cc_proto",
        ":scheduling_options_flags",
        ":scheduling_options_flags_cc_proto",
        "//xls/codegen:module_signature",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/logging:log_flags",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "simulate_module_main",
    srcs = ["simulate_module_main.cc"],
    deps = [
        ":eval_utils",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

py_test(
    name = "simulate_module_main_test",
    srcs = ["simulate_module_main_test.py"],
    data = [
        ":codegen_main",
        ":simulate_module_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    # iverilog crashes with ASAN and MSAN.
    tags = [
        "noasan",
        "nomsan",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

py_test(
    name = "codegen_main_test",
    srcs = ["codegen_main_test.py"],
    data = [
        ":codegen_main",
        "//xls/examples:sha256.opt.ir",
    ] + glob(["testdata/*.vtxt"]),
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = ["optonly"],
    deps = [
        "//xls/codegen:module_signature_py_pb2",
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@com_google_protobuf//:protobuf_python",
    ],
)

filegroup(
    name = "benchmark_test_sh",
    srcs = ["benchmark_test.sh"],
    visibility = ["//xls:xls_users"],
)

filegroup(
    name = "benchmark_eval_test_sh",
    srcs = ["benchmark_eval_test.sh"],
    visibility = ["//xls:xls_users"],
)

cc_binary(
    name = "benchmark_main",
    srcs = ["benchmark_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":codegen",
        ":codegen_flags",
        ":codegen_flags_cc_proto",
        ":scheduling_options_flags",
        ":scheduling_options_flags_cc_proto",
        "//xls/codegen:module_signature",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:binary_decision_diagram",
        "//xls/delay_model:analyze_critical_path",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:register",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:block_jit",
        "//xls/jit:function_jit",
        "//xls/jit:jit_channel_queue",
        "//xls/jit:jit_runtime",
        "//xls/jit:proc_jit",
        "//xls/passes:bdd_function",
        "//xls/passes:bdd_query_engine",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/passes:query_engine",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

py_test(
    name = "ir_minimizer_main_test",
    srcs = ["ir_minimizer_main_test.py"],
    data = ["ir_minimizer_main"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

# TODO(b/147004555): Add a JIT (or legacy) version of this test once
# the JIT supports enough functionality to pass.
py_test(
    name = "eval_ir_main_test",
    srcs = ["eval_ir_main_test.py"],
    data = [
        ":eval_ir_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "eval_ir_main_llvm_test",
    srcs = ["eval_ir_main_llvm_test.py"],
    data = [
        ":eval_ir_main",
        "@llvm-project//llvm:lli",
        "@llvm-project//llvm:llvm-link",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nomsan"],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "eval_proc_main_test",
    srcs = ["eval_proc_main_test.py"],
    data = [
        "testdata/eval_proc_main_test_block_memory.ir",
        "testdata/eval_proc_main_test_block_memory.sig.textproto",
        ":eval_proc_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

cc_binary(
    name = "solver",
    srcs = ["solver.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_binary(
    name = "bdd_stats",
    srcs = ["bdd_stats.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes:bdd_function",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "bom",
    srcs = ["bom.cc"],
    hdrs = ["bom.h"],
    deps = [
        "//xls/codegen:module_signature_cc_proto",
        "//xls/codegen:xls_metrics_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir:op",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "bom_test",
    srcs = ["bom_test.cc"],
    data = ["testdata/bom.sig.textproto"],
    deps = [
        ":bom",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_binary(
    name = "print_bom",
    srcs = ["print_bom.cc"],
    deps = [
        ":bom",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "cell_library_extract_formula",
    srcs = ["cell_library_extract_formula.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/netlist:lib_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "repl",
    srcs = ["repl.cc"],
    deps = [
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:mangle",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:scanner",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/dslx/type_system:concrete_type",
        "//xls/dslx/type_system:type_info",
        "//xls/dslx/type_system:typecheck_module",
        "//xls/ir",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_pass",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@linenoise",
    ],
)

cc_binary(
    name = "delay_info_main",
    srcs = ["delay_info_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/delay_model:analyze_critical_path",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:extract_stage",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "proto2bin",
    srcs = ["proto2bin_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/ir:ram_rewrite_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "benchmark_codegen_main",
    srcs = ["benchmark_codegen_main.cc"],
    deps = [
        ":codegen",
        ":codegen_flags",
        ":scheduling_options_flags",
        ":scheduling_options_flags_cc_proto",
        "//xls/codegen:block_metrics",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen:pipeline_generator",
        "//xls/codegen:xls_metrics_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

py_test(
    name = "benchmark_codegen_main_test",
    srcs = ["benchmark_codegen_main_test.py"],
    data = [
        ":benchmark_codegen_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "delay_info_main_test",
    srcs = ["delay_info_main_test.py"],
    data = [
        ":delay_info_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

py_binary(
    name = "run_timing_characterization",
    srcs = ["run_timing_characterization.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("portpicker"),
        "//xls/synthesis:timing_characterization_client",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
    ],
)

proto_library(
    name = "design_stats_proto",
    srcs = ["design_stats.proto"],
)

py_proto_library(
    name = "design_stats_py_pb2",
    deps = [":design_stats_proto"],
)

py_binary(
    name = "gather_design_stats",
    srcs = ["gather_design_stats.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":design_stats_py_pb2",
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "gather_design_stats_test",
    srcs = ["gather_design_stats_test.py"],
    data = [
        "testdata/find_index_5ps_model_unit_expected.textproto",
        "testdata/find_index_5ps_model_unit_verilog_sta_by_stage_sta.log",
        "testdata/find_index_5ps_model_unit_verilog_synth_by_stage_yosys_output.log.gz",
        ":gather_design_stats",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "delay_model_stats",
    srcs = ["delay_model_stats.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        "//xls/delay_model",
        "//xls/delay_model:delay_model_py_pb2",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)
