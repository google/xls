# Copyright 2024 The XLS Authors
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

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "apfloat_add_parallel_dslx",
    srcs = ["apfloat_add_parallel.x"],
    tags = ["optonly"],
)

xls_dslx_test(
    name = "apfloat_add_parallel_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":apfloat_add_parallel_dslx",
)

xls_dslx_opt_ir(
    name = "apfloat_add_parallel_opt_ir",
    srcs = ["apfloat_add_parallel.x"],
    dslx_top = "float32_top",
    ir_file = "apfloat_add_parallel.ir",
    opt_ir_file = "apfloat_add_parallel.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "apfloat_add_parallel_jit_wrapper",
    src = ":apfloat_add_parallel_opt_ir",
    jit_wrapper_args = {
        "class_name": "Float32AddSe",
        "namespace": "xls::examples",
    },
)

cc_test(
    name = "apfloat_add_parallel_test",
    srcs = ["apfloat_add_parallel_test.cc"],
    tags = ["optonly"],
    deps = [
        ":apfloat_add_parallel_jit_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
