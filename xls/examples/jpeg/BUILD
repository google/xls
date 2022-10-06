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
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_eval_ir_test",
    "xls_ir_verilog",
)
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load("@rules_hdl//synthesis:build_defs.bzl", "synthesize_rtl")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

xls_dslx_opt_ir(
    name = "idct_chen",
    srcs = ["idct_chen.x"],
    dslx_top = "idct",
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)

xls_ir_verilog(
    name = "idct_chen_5000ps_model_unit",
    src = ":idct_chen.opt.ir",
    codegen_args = {
        "delay_model": "unit",
        "clock_period_ps": "5000",
        "module_name": "idct_chen",
        "use_system_verilog": "false",
    },
    verilog_file = "idct_chen_5000ps_model_unit.v",
)

verilog_library(
    name = "idct_chen_5000ps_model_unit_verilog",
    srcs = [
        ":idct_chen_5000ps_model_unit.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "idct_chen_5000ps_model_unit_verilog_synth",
    tags = ["manual"],
    top_module = "idct_chen",
    deps = [
        ":idct_chen_5000ps_model_unit_verilog",
    ],
)

xls_dslx_test(
    name = "idct_chen_dslx_test",
    srcs = ["idct_chen.x"],
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test. Note: times out.
xls_eval_ir_test(
    name = "idct_chen_eval_ir_test",
    src = ":idct_chen.ir",
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)

cc_xls_ir_jit_wrapper(
    name = "idct_chen_jit_wrapper",
    src = ":idct_chen.opt.ir",
    jit_wrapper_args = {
        "class_name": "idct_chen",
        "namespace": "xls::jpeg",
    },
)

cc_test(
    name = "idct_chen_jit_wrapper_test",
    srcs = ["idct_chen_jit_wrapper_test.cc"],
    deps = [
        ":idct_chen_jit_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

# -- GRM

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "streams",
    srcs = ["streams.cc"],
    hdrs = ["streams.h"],
    deps = [
        ":constants",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "streams_test",
    srcs = ["streams_test.cc"],
    deps = [
        ":constants",
        ":streams",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "jpeg_grm",
    srcs = ["jpeg_grm.cc"],
    hdrs = ["jpeg_grm.h"],
    deps = [
        ":constants",
        ":idct_chen_jit_wrapper",
        ":streams",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "jpeg_grm_test",
    srcs = ["jpeg_grm_test.cc"],
    deps = [
        ":jpeg_grm",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)
