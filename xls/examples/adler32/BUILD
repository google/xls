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
    "xls_benchmark_ir",
    "xls_dslx_fmt_test",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_eval_ir_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_opt_ir(
    name = "adler32",
    srcs = ["adler32.x"],
    dslx_top = "main",
    ir_file = "adler32.ir",
    opt_ir_file = "adler32.opt.ir",
)

xls_dslx_test(
    name = "adler32_dslx_test",
    srcs = ["adler32.x"],
)

xls_dslx_fmt_test(
    name = "adler32_dslx_fmt_test",
    src = "adler32.x",
)

xls_eval_ir_test(
    name = "adler32_eval_ir_test",
    src = ":adler32.ir",
)

xls_benchmark_ir(
    name = "adler32_benchmark_ir",
    src = ":adler32.ir",
    benchmark_ir_args = {
        "delay_model": "sky130",
    },
)

cc_binary(
    name = "adler32_reference",
    srcs = ["adler32_reference.cc"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/logging",
    ],
)
