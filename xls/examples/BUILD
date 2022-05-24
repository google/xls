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

# Build rules for XLS examples.

load("//xls/examples:list_filegroup_files.bzl", "list_filegroup_files")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_eval_ir_test",
)
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load("@rules_hdl//synthesis:build_defs.bzl", "synthesize_rtl")
# TODO(cdleary): 2022-05-09 Re-enable use of this rule.
# load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

exports_files(["sha256.x"])

filegroup(
    name = "ir_examples",
    srcs = [
        ":adler32.ir",
        ":adler32.opt.ir",
        ":crc32.ir",
        ":crc32.opt.ir",
        ":matmul_4x4.ir",
        ":sha256.ir",
        ":sha256.opt.ir",
    ],
)

xls_dslx_opt_ir(
    name = "adler32",
    srcs = ["adler32.x"],
    dslx_top = "main",
    ir_file = "adler32.ir",
    opt_ir_file = "adler32.opt.ir",
)

xls_dslx_test(
    name = "adler32_dslx_test",
    srcs = ["adler32.x"],
)

# TODO(rspringer): OOMs. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "adler32_eval_ir_test",
    src = ":adler32.ir",
)

xls_benchmark_ir(
    name = "adler32_benchmark_ir",
    src = ":adler32.ir",
)

cc_binary(
    name = "adler32_reference",
    srcs = ["adler32_reference.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/logging",
    ],
)

xls_dslx_opt_ir(
    name = "capitalize",
    srcs = ["capitalize.x"],
    dslx_top = "main",
)

xls_dslx_opt_ir_test(
    name = "capitalize_test",
    dep = ":capitalize",
)

xls_dslx_opt_ir(
    name = "crc32",
    srcs = ["crc32.x"],
    dslx_top = "main",
    ir_file = "crc32.ir",
    opt_ir_file = "crc32.opt.ir",
)

xls_dslx_opt_ir_test(
    name = "crc32_test",
    dep = ":crc32",
)

cc_xls_ir_jit_wrapper(
    name = "crc32_jit_wrapper",
    src = ":crc32.opt.ir",
    jit_wrapper_args = {
        "class_name": "crc32",
    },
)

cc_binary(
    name = "crc32_reference",
    srcs = ["crc32_reference.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/logging",
    ],
)

xls_dslx_library(
    name = "dot_product_dslx",
    srcs = ["dot_product.x"],
    deps = [
        "//xls/modules:fp32_add_2_dslx",
        "//xls/modules:fp32_mul_2_dslx",
    ],
)

xls_dslx_test(
    name = "dot_product_dslx_test",
    srcs = ["dot_product.x"],
    dslx_test_args = {
        "compare": "none",
    },
    deps = [":dot_product_dslx"],
)

xls_dslx_library(
    name = "fir_filter_dslx",
    srcs = ["fir_filter.x"],
    deps = [
        "//xls/modules:fp32_add_2_dslx",
        "//xls/modules:fp32_mul_2_dslx",
    ],
)

xls_dslx_test(
    name = "fir_filter_dslx_test",
    srcs = ["fir_filter.x"],
    dslx_test_args = {
        "compare": "none",
    },
    deps = [":fir_filter_dslx"],
)

xls_dslx_test(
    name = "matmul_4x4_dslx_test",
    srcs = ["matmul_4x4.x"],
    # TODO(rspringer): 2022-05-24: Does not successfully lower to IR.
    # Needs support for arrays of channels.
    dslx_test_args = {
        "compare": "none",
    },
    deps = [
        "//xls/modules:fp32_mul_2",
    ],
)

xls_dslx_opt_ir(
    name = "riscv_simple",
    srcs = ["riscv_simple.x"],
    dslx_top = "run_instruction",
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test, xls_eval_ir_test and
# xls_benchmark_ir.
xls_dslx_test(
    name = "riscv_simple_dslx_test",
    srcs = ["riscv_simple.x"],
    dslx_test_args = {
        # TODO(hjmontero): run_instruction segfaults in the JIT.
        "compare": "interpreter",
    },
)

xls_dslx_opt_ir(
    name = "sha256",
    srcs = ["sha256.x"],
    dslx_top = "main",
    ir_file = "sha256.ir",
    opt_ir_file = "sha256.opt.ir",
    tags = ["optonly"],
)

xls_dslx_test(
    name = "sha256_dslx_test",
    srcs = ["sha256.x"],
)

# TODO(rspringer): OOMs on some machines. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "sha256_eval_ir_test",
    src = ":sha256.ir",
    tags = ["optonly"],
)

xls_benchmark_ir(
    name = "sha256_benchmark_ir",
    src = ":sha256.ir",
    tags = ["optonly"],
)

cc_xls_ir_jit_wrapper(
    name = "sha256_jit_wrapper",
    src = ":sha256.opt.ir",
    jit_wrapper_args = {
        "class_name": "sha256",
        "function": "main",
    },
)

xls_dslx_library(
    name = "sobel_filter_dslx",
    srcs = ["sobel_filter.x"],
    deps = [
        "//xls/modules:fp32_add_2_dslx",
        "//xls/modules:fp32_mul_2_dslx",
        "//third_party/xls_go_math:fpsqrt_32_dslx",
    ],
)

xls_dslx_test(
    name = "sobel_filter_test",
    srcs = ["sobel_filter.x"],
    deps = [":sobel_filter_dslx"],
)

list_filegroup_files(
    name = "ir_example_file_list",
    src = ":ir_examples",
    out = "ir_example_file_list.txt",
)

cc_library(
    name = "sample_packages",
    srcs = [
        "sample_packages.cc",
        "sample_packages.inc.h",
    ],
    hdrs = ["sample_packages.h"],
    data = [
        ":ir_example_file_list.txt",
        ":ir_examples",
        "//xls/examples:adler32.opt.ir",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
    ],
)

cc_library(
    name = "proc_fir_filter",
    srcs = ["proc_fir_filter.cc"],
    hdrs = ["proc_fir_filter.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:function_builder",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:value_helpers",
    ],
)

cc_test(
    name = "matmul_4x4_test",
    srcs = ["matmul_4x4_test.cc"],
    data = ["matmul_4x4.ir"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:init_xls",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:proc_network_interpreter",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "proc_fir_filter_test",
    srcs = ["proc_fir_filter_test.cc"],
    deps = [
        ":proc_fir_filter",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/interpreter:channel_queue",
        "//xls/interpreter:proc_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_scanner",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

xls_dslx_opt_ir(
    name = "proc_iota",
    srcs = ["proc_iota.x"],
    dslx_top = "main",
    ir_file = "proc_iota.ir",
    opt_ir_file = "proc_iota.opt.ir",
    tags = ["optonly"],
)

# -- find_index sample, with OpenROAD synthesis

xls_dslx_library(
    name = "find_index_dslx",
    srcs = ["find_index.x"],
)

xls_dslx_test(
    name = "find_index_test",
    library = ":find_index_dslx",
)

xls_dslx_verilog(
    name = "find_index_5000ps_model_unit",
    codegen_args = {
        "delay_model": "unit",
        "clock_period_ps": "5000",
        "module_name": "find_index",
        "use_system_verilog": "false",
    },
    dslx_top = "find_index",
    library = ":find_index_dslx",
    verilog_file = "find_index_5000ps_model_unit.v",
)

verilog_library(
    name = "find_index_5000ps_model_unit_verilog",
    srcs = [
        ":find_index_5000ps_model_unit.v",
    ],
)

synthesize_rtl(
    name = "find_index_5000ps_model_unit_verilog_synth",
    top_module = "find_index",
    deps = [
        ":find_index_5000ps_model_unit_verilog",
    ],
)

# TODO(cdleary): 2022-05-09 Get this target working as well, requires
# addressing the dependence on openmp runtime.
# place_and_route(
#     name = "find_index_place_and_route",
#     # ~64 MhZ
#     clock_period = "15.42857143",
#     core_padding_microns = 30,
#     placement_density = "0.8",
#     synthesized_rtl = ":find_index_5000ps_model_unit_verilog_synth",
#     target_die_utilization_percentage = "30",
# )
