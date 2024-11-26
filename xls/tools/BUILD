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

# pytype binary and test
load("@rules_python//python:proto.bzl", "py_proto_library")
# Load proto_library
# cc_proto_library is used in this file

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "node_coverage_stats_proto",
    srcs = ["node_coverage_stats.proto"],
    visibility = ["//xls:xls_users"],
    deps = ["//xls/ir:xls_value_proto"],
)

py_proto_library(
    name = "node_coverage_stats_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":node_coverage_stats_proto"],
)

cc_proto_library(
    name = "node_coverage_stats_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":node_coverage_stats_proto"],
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
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:type",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:function_extractor",
        "//xls/netlist:lib_parser",
        "//xls/netlist:netlist_cc_proto",
        "//xls/netlist:netlist_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/solvers:z3_lec",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@z3//:api",
    ],
)

cc_library(
    name = "proto_to_dslx",
    srcs = ["proto_to_dslx.cc"],
    hdrs = ["proto_to_dslx.h"],
    deps = [
        "//xls/common:math_util",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//src/google/protobuf/compiler:importer",
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
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "proto_to_dslx_test",
    srcs = ["proto_to_dslx_test.cc"],
    deps = [
        ":proto_to_dslx",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
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
        "@com_google_absl//absl/log:check",
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
        ":node_coverage_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:import_data",
        "//xls/dslx:mangle",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_kind",
        "//xls/dslx/ir_convert:conversion_info",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:observer",
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
        "//xls/jit:observer",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/tests:testvector_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/ir:xls_value_cc_proto",
        "//xls/tests:testvector_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
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
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "memory_models",
    srcs = ["memory_models.cc"],
    hdrs = ["memory_models.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/ir:ram_rewrite_cc_proto",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "eval_proc_main",
    srcs = ["eval_proc_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":eval_utils",
        ":memory_models",
        ":node_coverage_utils",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dev_tools:tool_timeout",
        "//xls/interpreter:block_evaluator",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:evaluator_options",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:serial_proc_runtime",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:block_elaboration",
        "//xls/ir:channel",
        "//xls/ir:channel_cc_proto",
        "//xls/ir:events",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:ram_rewrite_cc_proto",
        "//xls/ir:register",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:block_jit",
        "//xls/jit:jit_proc_runtime",
        "//xls/jit:jit_runtime",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "opt",
    srcs = ["opt.cc"],
    hdrs = ["opt.h"],
    visibility = ["//xls:xls_users"],
    deps = [
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:verifier",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/passes:pass_pipeline_cc_proto",
        "//xls/passes:verifier_checker",
        "@com_google_absl//absl/log",
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
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dev_tools:tool_timeout",
        "//xls/ir:ram_rewrite_cc_proto",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_pipeline_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

py_test(
    name = "opt_main_test",
    srcs = ["opt_main_test.py"],
    data = [
        ":opt_main",
        "//xls/modules/aes:aes_ctr.ir",
        "//xls/passes:default_optimization_pipeline_proto",
    ],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/passes:pass_pipeline_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "mul_folding_overlarge",
    srcs = ["run_opt_main.py"],
    args = ["xls/tools/testdata/mul_folding_overlarge.ir"],
    data = [
        ":opt_main",
        "//xls/tools/testdata:static_ir_files",
    ],
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
    data = [
        ":opt_main",
        "//xls/tools/testdata:static_ir_files",
    ],
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
    data = [
        ":opt_main",
        "//xls/tools/testdata:static_ir_files",
    ],
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
    data = [
        ":opt_main",
        "//xls/tools/testdata:static_ir_files",
    ],
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
    data = [
        ":opt_main",
        "//xls/tools/testdata:static_ir_files",
    ],
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

py_proto_library(
    name = "proc_channel_values_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":proc_channel_values_proto"],
)

cc_proto_library(
    name = "proc_channel_values_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":proc_channel_values_proto"],
)

proto_library(
    name = "scheduling_options_flags_proto",
    srcs = ["scheduling_options_flags.proto"],
    visibility = ["//xls:xls_users"],
    deps = ["@com_google_googleapis//google/api:field_behavior_proto"],
)

py_proto_library(
    name = "scheduling_options_flags_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":scheduling_options_flags_proto"],
)

cc_proto_library(
    name = "scheduling_options_flags_cc_proto",
    visibility = ["//xls:xls_users"],
    deps = [":scheduling_options_flags_proto"],
)

genrule(
    name = "default_codegen_and_scheduling_options",
    outs = [
        "default_codegen_options.textproto",
        "default_scheduling_options.textproto",
    ],
    cmd = """
    $(location :generate_options_protos_main) \\
          --codegen_options_used_textproto_file=$(location :default_codegen_options.textproto) \\
          --scheduling_options_used_textproto_file=$(location :default_scheduling_options.textproto)
    """,
    tools = [":generate_options_protos_main"],
    visibility = ["//xls:xls_users"],
)

cc_binary(
    name = "generate_options_protos_main",
    srcs = ["generate_options_protos_main.cc"],
    # Only for the genrule and xls-stuff to generate protos.
    visibility = ["//xls:xls_internal"],
    deps = [
        ":codegen_flags",
        ":scheduling_options_flags",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "scheduling_options_flags",
    srcs = ["scheduling_options_flags.cc"],
    hdrs = ["scheduling_options_flags.h"],
    deps = [
        ":scheduling_options_flags_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

proto_library(
    name = "codegen_flags_proto",
    srcs = ["codegen_flags.proto"],
    visibility = ["//xls:xls_users"],
    deps = ["//xls/ir:xls_ir_interface_proto"],
)

py_proto_library(
    name = "codegen_flags_py_pb2",
    visibility = ["//xls:xls_users"],
    deps = [":codegen_flags_proto"],
)

cc_proto_library(
    name = "codegen_flags_cc_proto",
    visibility = ["//xls:xls_users"],
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
        "//xls/ir:xls_ir_interface_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "codegen",
    srcs = ["codegen.cc"],
    hdrs = ["codegen.h"],
    deps = [
        ":codegen_flags_cc_proto",
        ":scheduling_options_flags_cc_proto",
        "//xls/codegen:codegen_options",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen:op_override_impls",
        "//xls/codegen:pipeline_generator",
        "//xls/codegen:ram_configuration",
        "//xls/codegen:unified_generator",
        "//xls/common:stopwatch",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/estimators/delay_model:ffi_delay_estimator",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/ir:verifier",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "//xls/scheduling:scheduling_pass",
        "//xls/scheduling:scheduling_pass_pipeline",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dev_tools:tool_timeout",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:verifier",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xls/codegen:module_signature_cc_proto",
        "//xls/codegen/vast",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulator",
        "//xls/simulation:verilog_simulators",
        "//xls/tests:testvector_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "//xls/tools/testdata:static_vtxt_files",
    ],
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
        ":node_coverage_stats_py_pb2",
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/ir:xls_value_py_pb2",
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
        ":eval_proc_main",
        "//xls/examples:delay.block.ir",
        "//xls/examples:delay.sig.textproto",
        "//xls/examples/dslx_module:manual_chan_caps_streaming_configured_multiproc.block.ir",
        "//xls/examples/dslx_module:manual_chan_caps_streaming_configured_multiproc.sig.textproto",
        "//xls/tools/testdata:eval_proc_main_conditional_test.opt.ir",
        "//xls/tools/testdata:eval_proc_main_test.opt.ir",
        "//xls/tools/testdata:eval_proc_main_test.ram_rewrites.textproto",
        "//xls/tools/testdata:eval_proc_main_test.sig.textproto",
        "//xls/tools/testdata:eval_proc_main_test.test_memory.block.ir",
        "//xls/tools/testdata:eval_proc_main_test.test_memory.ir",
        "//xls/tools/testdata:eval_proc_main_test.test_memory.opt.ir",
        "//xls/tools/testdata:eval_proc_main_test.test_memory.sig.textproto",
        "//xls/tools/testdata:eval_proc_main_test_broken.block.ir",
        "//xls/tools/testdata:eval_proc_main_test_with_trace.block.ir",
        "//xls/tools/testdata:eval_proc_main_zero_size_test.block.ir",
        "//xls/tools/testdata:eval_proc_main_zero_size_test.opt.ir",
        "//xls/tools/testdata:eval_proc_main_zero_size_test.sig.textproto",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":node_coverage_stats_py_pb2",
        ":proc_channel_values_py_pb2",
        "//xls/common:runfiles",
        "//xls/ir:xls_value_py_pb2",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
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
    data = ["//xls/tools/testdata:bom.sig.textproto"],
    deps = [
        ":bom",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
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
    name = "delay_info_main",
    srcs = ["delay_info_main.cc"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:analyze_critical_path",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/estimators/delay_model:delay_estimators",
        "//xls/estimators/delay_model:delay_info_cc_proto",
        "//xls/fdo:grpc_synthesizer",
        "//xls/fdo:synthesized_delay_diff_utils",
        "//xls/fdo:synthesizer",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "//xls/contrib/xlscc:hls_block_cc_proto",
        "//xls/ir:ram_rewrite_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
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

proto_library(
    name = "design_stats_proto",
    srcs = ["design_stats.proto"],
)

py_proto_library(
    name = "design_stats_py_pb2",
    visibility = ["//xls:xls_utility_users"],
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
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "gather_design_stats_test",
    srcs = ["gather_design_stats_test.py"],
    data = [
        ":gather_design_stats",
        "//xls/tools/testdata:find_index_5ps_model_unit_expected.textproto",
        "//xls/tools/testdata:find_index_5ps_model_unit_verilog_sta_by_stage_sta.log",
        "//xls/tools/testdata:find_index_5ps_model_unit_verilog_synth_by_stage_yosys_output.log.gz",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

cc_library(
    name = "node_coverage_utils",
    srcs = ["node_coverage_utils.cc"],
    hdrs = ["node_coverage_utils.h"],
    deps = [
        ":node_coverage_stats_cc_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:inline_bitmap",
        "//xls/data_structures:leaf_type_tree",
        "//xls/interpreter:observer",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/jit:jit_runtime",
        "//xls/jit:observer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)
