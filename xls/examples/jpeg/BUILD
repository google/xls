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

load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load("@rules_hdl//static_timing:build_defs.bzl", "run_opensta")
load("@rules_hdl//synthesis:build_defs.bzl", "synthesize_rtl")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_eval_ir_test",
    "xls_ir_verilog",
    "xls_ir_verilog_fdo",
    "xls_synthesis_metrics",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
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
    src = ":idct_chen",
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

synthesize_rtl(
    name = "idct_chen_5000ps_model_unit_verilog_synth_by_stage",
    synth_tcl = "//xls/synthesis/yosys:synth_by_stage.tcl",
    tags = ["manual"],
    top_module = "idct_chen",
    deps = [
        ":idct_chen_5000ps_model_unit_verilog",
    ],
)

xls_benchmark_ir(
    name = "idct_chen_13000ps_model_sky130_benchmark_ir",
    src = ":idct_chen.ir",
    benchmark_ir_args = {
        "delay_model": "sky130",
        "clock_period_ps": "13000",
    },
    tags = ["optonly"],
)

xls_ir_verilog(
    name = "idct_chen_13000ps_model_sky130",
    src = ":idct_chen",
    codegen_args = {
        "delay_model": "sky130",
        "clock_period_ps": "13000",
        "module_name": "idct_chen",
        "use_system_verilog": "false",
    },
    verilog_file = "idct_chen_13000ps_model_sky130.v",
)

verilog_library(
    name = "idct_chen_13000ps_model_sky130_verilog",
    srcs = [
        ":idct_chen_13000ps_model_sky130.v",
    ],
    tags = ["manual"],
)

synthesize_rtl(
    name = "idct_chen_13000ps_model_sky130_verilog_synth_by_stage",
    synth_tcl = "//xls/synthesis/yosys:synth_by_stage.tcl",
    tags = ["manual"],
    top_module = "idct_chen",
    deps = [
        ":idct_chen_13000ps_model_sky130_verilog",
    ],
)

run_opensta(
    name = "idct_chen_13000ps_model_sky130_verilog_sta_by_stage",
    sta_tcl = "//xls/synthesis/openroad:sta_by_stage.tcl",
    synth_target = ":idct_chen_13000ps_model_sky130_verilog_synth_by_stage",
    tags = ["manual"],
)

xls_synthesis_metrics(
    name = "idct_chen_13000ps_model_sky130_post_synth_metrics",
    srcs = [
        ":idct_chen_13000ps_model_sky130_verilog_sta_by_stage",
        ":idct_chen_13000ps_model_sky130_verilog_synth_by_stage",
    ],
    tags = ["manual"],
)

xls_ir_verilog_fdo(
    name = "idct_chen_6400ps_model_sky130_fdo",
    src = ":idct_chen",
    codegen_args = {
        "delay_model": "sky130",
        "clock_period_ps": "6400",
        "module_name": "idct_chen",
        "use_system_verilog": "false",
        # Activate FDO
        "use_fdo": "true",
        # The following parameters can optionally be specified
        "fdo_iteration_number": "5",
        "fdo_delay_driven_path_number": "1",
        "fdo_fanout_driven_path_number": "0",
        "fdo_refinement_stochastic_ratio": "1.0",
        "fdo_path_evaluate_strategy": "window",
    },
    # Must specify standard_cells if using FDO, if not using the default sky130
    standard_cells = "@com_google_skywater_pdk_sky130_fd_sc_hd//:sky130_fd_sc_hd",
    verilog_file = "idct_chen_6400ps_model_sky130_fdo.v",
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
    src = ":idct_chen",
    jit_wrapper_args = {
        "class_name": "IdctChen",
        "namespace": "xls::jpeg",
    },
)

cc_test(
    name = "idct_chen_jit_wrapper_test",
    srcs = ["idct_chen_jit_wrapper_test.cc"],
    deps = [
        ":idct_chen_jit_wrapper",
        "@com_google_googletest//:gtest",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:value_view",
    ],
)

cc_test(
    name = "jpeg_grm_test",
    srcs = ["jpeg_grm_test.cc"],
    deps = [
        ":jpeg_grm",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
