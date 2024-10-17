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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_fmt_test",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "assertions_dslx",
    srcs = ["assertions.x"],
)

xls_dslx_fmt_test(
    name = "assertions_fmt_test",
    src = "assertions.x",
)

xls_dslx_test(
    name = "assertions_dslx_test",
    srcs = ["assertions.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_ir(
    name = "assertions_ir",
    dslx_top = "main",
    ir_file = "assertions.ir",
    library = ":assertions_dslx",
)

xls_ir_opt_ir(
    name = "assertions_opt_ir",
    src = "assertions.ir",
)

xls_ir_verilog(
    name = "assertions_comb_sv",
    src = ":assertions_opt_ir",
    codegen_args = {
        "module_name": "assertions_top",
        "generator": "combinational",
        "delay_model": "unit",
        "use_system_verilog": "true",
    },
    verilog_file = "assertions_comb.sv",
)

xls_ir_verilog(
    name = "assertions_4_stages_sv",
    src = ":assertions_opt_ir",
    codegen_args = {
        "module_name": "assertions_top",
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
        "reset_data_path": "false",
        "use_system_verilog": "true",
    },
    verilog_file = "assertions_4_stages.sv",
)

cc_test(
    name = "assertions_test",
    srcs = ["assertions_test.cc"],
    data = [
        "assertions_4_stages.sig.textproto",
        "assertions_4_stages.sv",
        "assertions_comb.sig.textproto",
        "assertions_comb.sv",
    ],
    deps = [
        "//xls/codegen:module_signature",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/codegen/vast",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/simulation:default_verilog_simulator",
        "//xls/simulation:module_simulator",
        "//xls/simulation:verilog_simulator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)
