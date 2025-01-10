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

# Build rules for proto to DSLX example

# Load proto_library
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load("//xls/build_rules:xls_utilities.bzl", "proto_to_dslx")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

proto_library(
    name = "proto_schema_lib",
    srcs = [":param.proto"],
)

proto_to_dslx(
    name = "parameters",
    proto_name = "xls.DesignVariants",
    proto_schema = ":proto_schema_lib",
    textproto = ":parameters.textproto",
)

xls_dslx_library(
    name = "parameters_lib",
    srcs = [":parameters"],
)

xls_dslx_library(
    name = "design_lib",
    srcs = [
        "design.x",
    ],
    deps = [":parameters_lib"],
)

xls_dslx_ir(
    name = "design_ir",
    dslx_top = "design_top",
    ir_file = "design.ir",
    library = "design_lib",
)

xls_ir_opt_ir(
    name = "design_opt_ir",
    src = "design.ir",
    top = "__design__design_top",
)

xls_ir_verilog(
    name = "design_verilog",
    src = ":design_opt_ir",
    codegen_args = {
        "module_name": "design_top",
        "delay_model": "unit",
        "pipeline_stages": "1",
        "use_system_verilog": "false",
    },
    verilog_file = "design.v",
)
