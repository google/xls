# Copyright 2020 Google LLC
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

load("//xls/build:build_defs.bzl", "dslx_test")
load("//xls/examples:list_filegroup_files.bzl", "list_filegroup_files")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

filegroup(
    name = "ir_examples",
    srcs = [
        ":adler32_all_ir",
        ":crc32_all_ir",
        ":sha256_all_ir",
    ],
)

dslx_test(
    name = "adler32",
    srcs = ["adler32.x"],
    # TODO(rspringer): OOMs.
    prove_unopt_eq_opt = False,
)

cc_binary(
    name = "adler32_reference",
    srcs = ["adler32_reference.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common:integral_types",
        "//xls/common/logging",
    ],
)

dslx_test(
    name = "crc32",
    srcs = ["crc32.x"],
)

cc_binary(
    name = "crc32_reference",
    srcs = ["crc32_reference.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common:integral_types",
        "//xls/common/logging",
    ],
)

dslx_test(
    name = "riscv_simple",
    srcs = ["riscv_simple.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "sha256",
    srcs = ["sha256.x"],
    # TODO(rspringer): OOMs on some machines.
    prove_unopt_eq_opt = False,
    tags = ["optonly"],
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
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
    ],
)
