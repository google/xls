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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:abstract_node_evaluator",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:type",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir:ir_test_base",
        "//xls/jit:ir_jit",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "booleanify_main",
    srcs = ["booleanify_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":booleanifier",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/codegen:vast",
    ],
)

cc_library(
    name = "io_strategy_factory",
    srcs = ["io_strategy_factory.cc"],
    hdrs = ["io_strategy_factory.h"],
    deps = [
        ":io_strategy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "ir_stats_main",
    srcs = ["ir_stats_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
    ],
)

cc_binary(
    name = "check_ir_equivalence_main",
    srcs = ["check_ir_equivalence_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes",
        "//xls/passes:dce_pass",
        "//xls/passes:inlining_pass",
        "//xls/passes:map_inlining_pass",
        "//xls/passes:pass_base",
        "//xls/passes:unroll_pass",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:extract_stage",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
    ],
)

cc_binary(
    name = "lec_main",
    srcs = ["lec_main.cc"],
    data = [
        "//xls/dslx:ir_converter_main",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:init_xls",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:function_extractor",
        "//xls/netlist:lib_parser",
        "//xls/netlist:netlist_cc_proto",
        "//xls/netlist:netlist_parser",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_lec",
        "//xls/solvers:z3_netlist_translator",
        "//xls/solvers:z3_utils",
        "@z3//:api",
    ],
)

cc_binary(
    name = "netlist_interpreter_main",
    srcs = ["netlist_interpreter_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/codegen:flattening",
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
    ],
)

cc_library(
    name = "proto_to_dslx",
    srcs = ["proto_to_dslx.cc"],
    hdrs = ["proto_to_dslx.h"],
    deps = [
        # protobuf compiler import
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:ast",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "proto_to_dslx_main",
    srcs = ["proto_to_dslx_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":proto_to_dslx",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "proto_to_dslx_test",
    srcs = ["proto_to_dslx_test.cc"],
    deps = [
        ":proto_to_dslx",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_directory",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "smtlib_emitter_main",
    srcs = ["smtlib_emitter_main.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:abstract_node_evaluator",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
        "@z3//:api",
    ],
)

cc_library(
    name = "testbench",
    hdrs = ["testbench.h"],
    deps = [
        ":testbench_thread",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "//xls/common/file:filesystem",
    ],
)

cc_library(
    name = "testbench_builder",
    hdrs = ["testbench_builder.h"],
    deps = [
        ":testbench",
        ":testbench_builder_helpers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "testbench_builder_helpers",
    hdrs = ["testbench_builder_helpers.h"],
    deps = [
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "testbench_thread",
    hdrs = ["testbench_thread.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xls/common:thread",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/jit:ir_jit",
    ],
)

cc_library(
    name = "wrap_io",
    srcs = ["wrap_io.cc"],
    hdrs = ["wrap_io.h"],
    deps = [
        ":io_strategy",
        "@com_google_absl//absl/status:statusor",
        "//xls/codegen:finite_state_machine",
        "//xls/codegen:module_signature",
        "//xls/codegen:vast",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:type",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/simulation:module_testbench",
        "//xls/simulation:verilog_simulators",
        "//xls/simulation:verilog_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ice40_io_strategy",
    srcs = ["ice40_io_strategy.cc"],
    hdrs = ["ice40_io_strategy.h"],
    data = ["//xls/uncore_rtl/ice40:iceprog_includes"],
    deps = [
        ":wrap_io",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/codegen:vast",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "//xls/codegen:module_signature",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:init_xls",
        "//xls/common/logging",
        "//xls/ir:ir_parser",
    ],
)

cc_binary(
    name = "parse_ir",
    srcs = ["parse_ir.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
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
    name = "eval_ir_main",
    srcs = ["eval_ir_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:ir_converter",
        "//xls/dslx:mangle",
        "//xls/dslx:parse_and_typecheck",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir:ir_parser",
        "//xls/ir:random_value",
        "//xls/jit:ir_jit",
        "//xls/passes",
        "//xls/passes:standard_pipeline",
    ],
)

cc_library(
    name = "device_rpc_strategy",
    hdrs = ["device_rpc_strategy.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/ir:type",
        "//xls/ir:value",
    ],
)

cc_library(
    name = "device_rpc_strategy_factory",
    srcs = ["device_rpc_strategy_factory.cc"],
    hdrs = ["device_rpc_strategy_factory.h"],
    deps = [
        ":device_rpc_strategy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ice40_device_rpc_strategy",
    srcs = ["ice40_device_rpc_strategy.cc"],
    hdrs = ["ice40_device_rpc_strategy.h"],
    deps = [
        ":device_rpc_strategy",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:math_util",
        "//xls/common:strerror",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/codegen:vast",
    ],
)

cc_library(
    name = "opt",
    srcs = ["opt.cc"],
    hdrs = ["opt.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/dslx:ir_converter",
        "//xls/dslx:parse_and_typecheck",
        "//xls/ir:ir_parser",
        "//xls/passes",
        "//xls/passes:standard_pipeline",
    ],
)

cc_binary(
    name = "opt_main",
    srcs = ["opt_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        ":opt",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes",
        "//xls/passes:standard_pipeline",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:init_xls",
        "//xls/common:subprocess",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "//xls/ir:number_parser",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
        "//xls/jit:ir_jit",
        "//xls/passes",
        "//xls/passes:arith_simplification_pass",
        "//xls/passes:array_simplification_pass",
        "//xls/passes:bit_slice_simplification_pass",
        "//xls/passes:concat_simplification_pass",
        "//xls/passes:constant_folding_pass",
        "//xls/passes:cse_pass",
        "//xls/passes:dce_pass",
        "//xls/passes:dfe_pass",
        "//xls/passes:inlining_pass",
        "//xls/passes:standard_pipeline",
        "//xls/passes:tuple_simplification_pass",
        "//xls/passes:unroll_pass",
    ],
)

cc_binary(
    name = "llvm_main_generator",
    srcs = ["llvm_main_generator.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/jit:jit_runtime",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
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
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
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
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
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
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
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
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
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
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
    ],
)

cc_binary(
    name = "codegen_main",
    srcs = ["codegen_main.cc"],
    visibility = ["//xls:xls_users"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/codegen:pipeline_generator",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir:ir_parser",
        "//xls/passes:standard_pipeline",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_pass",
    ],
)

cc_binary(
    name = "simulate_module_main",
    srcs = ["simulate_module_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/codegen:module_signature",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:format_preference",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulators",
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
    ],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = ["optonly"],
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "//xls/codegen:module_signature_py_pb2",
        "//xls/common:runfiles",
        "//xls/common:test_base",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//xls/codegen:module_signature",
        "//xls/codegen:pipeline_generator",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/delay_model:analyze_critical_path",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes",
        "//xls/passes:bdd_query_engine",
        "//xls/passes:standard_pipeline",
        "//xls/scheduling:pipeline_schedule",
    ],
)

py_test(
    name = "ir_minimizer_main_test",
    srcs = ["ir_minimizer_main_test.py"],
    data = ["ir_minimizer_main"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common:runfiles",
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
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "//xls/ir/python:ir_parser",
    ],
)

cc_binary(
    name = "solver",
    srcs = ["solver.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
    ],
)

cc_binary(
    name = "bdd_stats",
    srcs = ["bdd_stats.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes:bdd_function",
    ],
)

cc_binary(
    name = "cell_library_extract_formula",
    srcs = ["cell_library_extract_formula.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/netlist:lib_parser",
    ],
)

cc_binary(
    name = "repl",
    srcs = ["repl.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/codegen:combinational_generator",
        "//xls/codegen:module_signature",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/codegen:pipeline_generator",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/logging:log_message",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/dslx:ast",
        "//xls/dslx:builtins",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:concrete_type",
        "//xls/dslx:error_printer",
        "//xls/dslx:import_data",
        "//xls/dslx:interpreter",
        "//xls/dslx:ir_converter",
        "//xls/dslx:mangle",
        "//xls/dslx:parser",
        "//xls/dslx:scanner",
        "//xls/dslx:type_info",
        "//xls/dslx:typecheck",
        "//xls/ir",
        "//xls/passes:standard_pipeline",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_pass",
        "@linenoise",
    ],
)

cc_binary(
    name = "delay_info_main",
    srcs = ["delay_info_main.cc"],
    deps = [
        ":opt",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/delay_model:analyze_critical_path",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/passes",
        "//xls/passes:standard_pipeline",
        "//xls/scheduling:extract_stage",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
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
