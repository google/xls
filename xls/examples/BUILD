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
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_module_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_eval_ir_test",
    "xls_ir_jit_wrapper",
)

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

xls_dslx_module_library(
    name = "adler32_dslx_module",
    src = "adler32.x",
)

xls_dslx_opt_ir(
    name = "adler32",
    dep = ":adler32_dslx_module",
    ir_file = "adler32.ir",
    opt_ir_file = "adler32.opt.ir",
)

xls_dslx_test(
    name = "adler32_dslx_test",
    dep = ":adler32_dslx_module",
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

xls_dslx_module_library(
    name = "capitalize_dslx_module",
    src = "capitalize.x",
)

xls_dslx_opt_ir(
    name = "capitalize",
    dep = ":capitalize_dslx_module",
)

xls_dslx_opt_ir_test(
    name = "capitalize_test",
    dep = ":capitalize",
)

xls_dslx_module_library(
    name = "crc32_dslx_module",
    src = "crc32.x",
)

xls_dslx_opt_ir(
    name = "crc32",
    dep = ":crc32_dslx_module",
    ir_file = "crc32.ir",
    opt_ir_file = "crc32.opt.ir",
)

xls_dslx_opt_ir_test(
    name = "crc32_test",
    dep = ":crc32",
)

xls_ir_jit_wrapper(
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
        "//xls/modules:fpadd_2x32_dslx",
        "//xls/modules:fpmul_2x32_dslx",
    ],
)

xls_dslx_module_library(
    name = "dot_product_dslx_module",
    src = "dot_product.x",
    deps = [":dot_product_dslx"],
)

xls_dslx_test(
    name = "dot_product_dslx_test",
    dep = ":dot_product_dslx_module",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_library(
    name = "fir_filter_dslx",
    srcs = ["fir_filter.x"],
    deps = [
        "//xls/modules:fpadd_2x32_dslx",
        "//xls/modules:fpmul_2x32_dslx",
    ],
)

xls_dslx_module_library(
    name = "fir_filter_dslx_module",
    src = "fir_filter.x",
    deps = [":fir_filter_dslx"],
)

xls_dslx_test(
    name = "fir_filter_dslx_test",
    dep = ":fir_filter_dslx_module",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_module_library(
    name = "riscv_simple_dslx_module",
    src = "riscv_simple.x",
)

xls_dslx_opt_ir(
    name = "riscv_simple",
    dep = ":riscv_simple_dslx_module",
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test, xls_eval_ir_test and
# xls_benchmark_ir.
xls_dslx_test(
    name = "riscv_simple_dslx_test",
    dep = ":riscv_simple_dslx_module",
    dslx_test_args = {
        # TODO(hjmontero): run_instruction segfaults in the JIT.
        "compare": "interpreter",
    },
)

xls_dslx_module_library(
    name = "sha256_dslx_module",
    src = "sha256.x",
)

xls_dslx_opt_ir(
    name = "sha256",
    dep = ":sha256_dslx_module",
    ir_file = "sha256.ir",
    opt_ir_file = "sha256.opt.ir",
    tags = ["optonly"],
)

xls_dslx_test(
    name = "sha256_dslx_test",
    dep = ":sha256_dslx_module",
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

xls_ir_jit_wrapper(
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
        "//xls/modules:fpadd_2x32_dslx",
        "//xls/modules:fpmul_2x32_dslx",
        "//third_party/xls_go_math:fpsqrt_32_dslx",
    ],
)

xls_dslx_module_library(
    name = "sobel_filter_dslx_module",
    src = "sobel_filter.x",
    deps = [":sobel_filter_dslx"],
)

xls_dslx_test(
    name = "sobel_filter_test",
    dep = ":dot_product_dslx_module",
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
