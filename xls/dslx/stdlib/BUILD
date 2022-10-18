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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "get_mangled_ir_symbol",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "std_dslx",
    srcs = ["std.x"],
)

xls_dslx_test(
    name = "std_dslx_test",
    srcs = ["std.x"],
)

xls_dslx_test(
    name = "acm_random_dslx_test",
    srcs = ["acm_random.x"],
)

xls_dslx_library(
    name = "apfloat_dslx",
    srcs = ["apfloat.x"],
    deps = [":std_dslx"],
)

xls_dslx_test(
    name = "apfloat_dslx_test",
    srcs = ["apfloat.x"],
    deps = [":apfloat_dslx"],
)

xls_dslx_library(
    name = "bfloat16_dslx",
    srcs = ["bfloat16.x"],
    deps = [":std_dslx"],
)

xls_dslx_test(
    name = "bfloat16_dslx_test",
    srcs = ["bfloat16.x"],
    deps = [":bfloat16_dslx"],
)

xls_dslx_library(
    name = "float32_dslx",
    srcs = ["float32.x"],
    deps = [":std_dslx"],
)

xls_dslx_test(
    name = "float32_dslx_test",
    srcs = ["float32.x"],
    deps = [":float32_dslx"],
)

xls_dslx_library(
    name = "float64_dslx",
    srcs = ["float64.x"],
    deps = [":std_dslx"],
)

xls_dslx_test(
    name = "float64_dslx_test",
    srcs = ["float64.x"],
    deps = [":float64_dslx"],
)

xls_dslx_opt_ir(
    name = "float32_to_int32",
    srcs = ["float32.x"],
    dslx_top = "to_int32",
    deps = [
        ":apfloat_dslx",
    ],
)

cc_xls_ir_jit_wrapper(
    name = "float32_to_int32_wrapper",
    src = ":float32_to_int32.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32ToInt32",
        "function": get_mangled_ir_symbol("float32", "to_int32"),
        "namespace": "xls::dslx",
    },
)

xls_dslx_opt_ir(
    name = "float32_from_int32",
    srcs = ["float32.x"],
    dslx_top = "from_int32",
    deps = [
        ":apfloat_dslx",
    ],
)

cc_xls_ir_jit_wrapper(
    name = "float32_from_int32_wrapper",
    src = ":float32_from_int32.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32FromInt32",
        "function": get_mangled_ir_symbol("float32", "from_int32"),
        "namespace": "xls::dslx",
    },
)

cc_test(
    name = "float32_test",
    srcs = ["float32_test.cc"],
    tags = ["optonly"],
    deps = [
        ":float32_from_int32_wrapper",
        ":float32_to_int32_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_users"],
)
