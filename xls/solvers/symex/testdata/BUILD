# Copyright 2026 The XLS Authors
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
    "xls_dslx_ir",
    "xls_dslx_library",
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

xls_dslx_library(
    name = "execute_alu_dslx",
    srcs = ["execute_alu.x"],
)

xls_dslx_ir(
    name = "execute_alu_ir",
    dslx_top = "execute_alu",
    ir_file = "execute_alu.ir",
    library = ":execute_alu_dslx",
)

xls_dslx_library(
    name = "three_way_compare_dslx",
    srcs = ["three_way_compare.x"],
)

xls_dslx_ir(
    name = "three_way_compare_ir",
    dslx_top = "three_way_compare",
    ir_file = "three_way_compare.ir",
    library = ":three_way_compare_dslx",
)

xls_dslx_library(
    name = "priority_encoder_dslx",
    srcs = ["priority_encoder.x"],
)

xls_dslx_ir(
    name = "priority_encoder_ir",
    dslx_top = "priority_encoder",
    ir_file = "priority_encoder.ir",
    library = ":priority_encoder_dslx",
)
