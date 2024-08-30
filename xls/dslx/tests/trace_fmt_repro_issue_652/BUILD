# Copyright 2023 The XLS Authors
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
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "trace_fmt_repro_dslx",
    srcs = ["trace_fmt_repro.x"],
)

xls_dslx_opt_ir(
    name = "trace_fmt_repro",
    dslx_top = "foo",
    # Don't emit asserts until they are emitted in a format which is expected by
    # the project consuming the Verilog.
    ir_conv_args = {
        "emit_fail_as_assert": "false",
    },
    library = ":trace_fmt_repro_dslx",
)

xls_dslx_opt_ir(
    name = "trace_fmt_repro_no_conv_args",
    dslx_top = "foo",
    library = ":trace_fmt_repro_dslx",
)

xls_dslx_test(
    name = "trace_fmt_repro_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":trace_fmt_repro_dslx",
)
