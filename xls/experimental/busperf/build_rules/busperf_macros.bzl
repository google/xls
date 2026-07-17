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

"""Macros wiring up a busperf ready/valid setup for an XLS proc. Private - load from build_defs.bzl."""

load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_verilog")
load(":busperf_report_rules.bzl", "busperf_analyze")
load(":busperf_yaml_rules.bzl", "busperf_yaml")

visibility(["//xls/experimental/busperf/build_rules/..."])

def xls_busperf_yaml(
        name,
        dslx_top,
        library,
        reset,
        scope,
        reset_active_low = False,
        codegen_args = {}):
    """Generates a busperf YAML bus description for one XLS proc design.

    Args:
      name: Base name for the generated targets and Verilog module.
      dslx_top: DSLX proc name to codegen as the top of the design.
      library: xls_dslx_library target containing dslx_top.
      reset: Reset signal name.
      scope: Dot-separated VCD scope path to the DUT, e.g. "tb_foo.dut".
      reset_active_low: Reset polarity. Defaults to False.
      codegen_args: Extra/override codegen_main args.
    """
    final_codegen_args = dict(codegen_args)
    final_codegen_args["reset"] = reset
    final_codegen_args["reset_active_low"] = "true" if reset_active_low else "false"

    xls_dslx_verilog(
        name = name + "_verilog",
        dslx_top = dslx_top,
        library = library,
        codegen_args = final_codegen_args,
        verilog_file = name + ".v",
        tags = ["manual"],
    )

    busperf_yaml(
        name = name + "_bus_yaml",
        signature = ":" + name + ".sig.textproto",
        scope = scope,
        xls_sig_to_busperf = "//xls/experimental/busperf:xls_sig_to_busperf",
        tags = ["manual"],
    )

def xls_busperf_setup(
        name,
        dslx_top,
        library,
        testbench,
        reset,
        scope,
        vcd_filename,
        reset_active_low = False,
        testbench_defines = {},
        codegen_args = {}):
    """Generates a full busperf channel analysis setup for one XLS proc design.

    Args:
      name: Base name for the generated targets and Verilog module.
      dslx_top: DSLX proc name to codegen as the top of the design.
      library: xls_dslx_library target containing dslx_top.
      testbench: Checked-in Verilog testbench.
      reset: Reset signal name.
      scope: Dot-separated VCD scope path to the DUT, e.g. "tb_foo.dut".
      vcd_filename: VCD file name
      reset_active_low: Reset polarity. Defaults to False.
      testbench_defines: defines to pass to iverilog simulator
      codegen_args: Extra/override codegen_main args.
    """
    # `.v` is hardcoded throughout this macro (the `_vcd` genrule).
    setup_codegen_args = dict(codegen_args)
    setup_codegen_args["use_system_verilog"] = "false"

    xls_busperf_yaml(
        name = name,
        dslx_top = dslx_top,
        library = library,
        reset = reset,
        scope = scope,
        reset_active_low = reset_active_low,
        codegen_args = setup_codegen_args,
    )

    testbench_define_flags = " ".join([
        "-D'{}={}'".format(k, v)
        for k, v in testbench_defines.items()
    ])

    native.genrule(
        name = name + "_vcd",
        srcs = [":" + name + ".v", testbench],
        outs = [name + ".vcd"],
        cmd = (
            "$(location @com_icarus_iverilog//:iverilog) -g2012 {defines} " +
            "-o sim.vvp $(location :{design_v}) $(location {tb_v}) && " +
            "$(location @com_icarus_iverilog//:vvp) sim.vvp && " +
            "mv {vcd_filename} $@"
        ).format(
            defines = testbench_define_flags,
            design_v = name + ".v",
            tb_v = testbench,
            vcd_filename = vcd_filename,
        ),
        tools = [
            "@com_icarus_iverilog//:iverilog",
            "@com_icarus_iverilog//:vvp",
        ],
        tags = ["manual"],
    )

    for suffix, format in [("_stats", "text"), ("_report", "html")]:
        busperf_analyze(
            name = name + suffix,
            vcd = ":" + name + "_vcd",
            bus_yaml = ":" + name + "_bus_yaml",
            format = format,
            busperf_bin = "@busperf//:busperf_bin",
            tags = ["manual"],
        )

    native.genrule(
        name = name + "_open_report",
        srcs = [":" + name + "_report"],
        outs = [name + "_open_report.sh"],
        cmd = ("""
cat > $@ <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
report="bazel-bin/{package}/{report_file}"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$$report"
elif command -v open >/dev/null 2>&1; then
  open "$$report"
else
  echo "No browser opener found; report is at: $$report" >&2
  exit 1
fi
EOF
chmod +x $@
""").format(
            package = native.package_name(),
            report_file = name + "_report.report.html",
        ),
        executable = True,
        tags = ["manual"],
    )
