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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_dslx_verilog",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "passthrough_dslx",
    srcs = [
        "passthrough.x",
    ],
)

xls_dslx_opt_ir(
    name = "pasthrough_opt_ir",
    dslx_top = "Passthrough",
    library = "passthrough_dslx",
)

xls_dslx_verilog(
    name = "passthrough_verilog",
    codegen_args = {
        "module_name": "passthrough",
        "delay_model": "unit",
        "pipeline_stages": "1",
        "reset": "rst",
        "use_system_verilog": "false",
    },
    dslx_top = "Passthrough",
    library = "passthrough_dslx",
    verilog_file = "passthrough.v",
)

xls_dslx_test(
    name = "passthrough_dslx_ir_test",
    dslx_test_args = {
        "compare": "interpreter",
    },
    library = "passthrough_dslx",
)
