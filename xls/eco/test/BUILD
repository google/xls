# Copyright 2025 The XLS Authors
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

"""Tests for the ECO workflow."""

load("//xls/eco:eco_build_defs.bzl", "xls_dslx_ir_diff", "xls_patch_ir")

package(default_visibility = ["//xls:xls_internal"])

xls_dslx_ir_diff(
    name = "crc32_diff",
    srcs = [
        ":crc32.org.x",
        ":crc32.rev.x",
    ],
    dslx_top = "main",
    mcs = False,
    verbosity = 1,
)

xls_dslx_ir_diff(
    name = "apfloat_diff0",
    srcs = [
        ":apfloat_fmac.org.x",
        ":apfloat_fmac.rev0.x",
    ],
    dslx_top = "fmac",
    mcs = True,
    verbosity = 1,
)

xls_dslx_ir_diff(
    name = "apfloat_diff1",
    srcs = [
        ":apfloat_fmac.org.x",
        ":apfloat_fmac.rev1.x",
    ],
    dslx_top = "fmac",
    mcs = False,
    verbosity = 1,
)

xls_dslx_ir_diff(
    name = "riscv_diff",
    srcs = [
        ":riscv_simple.org.x",
        ":riscv_simple.rev.x",
    ],
    dslx_top = "run_instruction",
    mcs = False,
    verbosity = 1,
)

xls_dslx_ir_diff(
    name = "histogram_diff",
    srcs = [
        ":histogram.org.x",
        ":histogram.rev.x",
    ],
    dslx_top = "histogram",
    mcs = False,
    verbosity = 1,
)

#A tricky one Takes about 50 minutes to run.
xls_dslx_ir_diff(
    name = "vector_core_diff",
    srcs = [
        ":vector_core.org.x",
        ":vector_core.rev.x",
    ],
    dslx_top = "vector_core",
    mcs = False,
    verbosity = 1,
)

xls_dslx_ir_diff(
    name = "fir_filter_fp_diff",
    srcs = [
        ":fir_filter_fp32.org.x",
        ":fir_filter_fp32.rev.x",
    ],
    dslx_top = "main",
    mcs = True,
    mcs_cutoff = 1500,
    verbosity = 1,
)

xls_dslx_ir_diff(
    name = "fir_filter_fixed32_diff",
    srcs = [
        ":fir_filter_fixed32.org.x",
        ":fir_filter_fixed32.rev.x",
    ],
    dslx_top = "main",
    mcs = True,
    verbosity = 1,
)

xls_patch_ir(
    name = "crc32_patched_opt_ir",
    ir_diff = ":crc32_diff",
)

#This will fail due to not supporting changes in channel types yet, needs manual fixup.
# 1. replace the header channel defs  in the patched ir with the after ir ones.
# 2. replace top receive nodes with the after ir ones while keeping their ids
# 3. run lec manually and it should pass.
# 4. TODO: Treat channles as nodes and automatically re-generate channels if needed.
xls_patch_ir(
    name = "apfloat_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":apfloat_diff_rev1",
)

xls_patch_ir(
    name = "riscv_patched_opt_ir",
    ir_diff = ":riscv_diff",
)

xls_patch_ir(
    name = "histogram_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":histogram_diff",
)

# TODO: Find the sweet spot to hand-off MCS to GED to minimize the runtime.
xls_patch_ir(
    name = "vector_core_patched_opt_ir",
    ir_diff = ":vector_core_diff",
)

xls_patch_ir(
    name = "fir_filter_patched_opt_ir",
    ir_diff = ":fir_filter_fp_diff",
)

xls_dslx_ir_diff(
    name = "fir_filter_fixed_diff",
    srcs = [
        ":fir_filter_fixed32.org.x",
        ":fir_filter_fixed32.rev.x",
    ],
    dslx_top = "main",
    mcs = True,
    verbosity = 1,
)

xls_patch_ir(
    name = "fir_filter_fixed_patched_opt_ir",
    ir_diff = ":fir_filter_fixed_diff",
)
