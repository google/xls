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

"""Rule that runs `busperf analyze` to produce a text or HTML report."""

visibility(["//xls/experimental/busperf/build_rules/..."])

_FORMAT_TO_FLAG_AND_EXT = {
    "text": ("--text", ".stats.txt"),
    "html": ("--html", ".report.html"),
}

def _busperf_analyze_impl(ctx):
    flag, ext = _FORMAT_TO_FLAG_AND_EXT[ctx.attr.format]
    out = ctx.actions.declare_file(ctx.label.name + ext)

    args = ctx.actions.args()
    args.add("analyze")
    args.add(ctx.file.vcd)
    args.add(ctx.file.bus_yaml)
    args.add(flag)
    args.add("-o", out)

    ctx.actions.run(
        executable = ctx.executable.busperf_bin,
        arguments = [args],
        inputs = [ctx.file.vcd, ctx.file.bus_yaml],
        outputs = [out],
        mnemonic = "BusperfAnalyze",
        progress_message = "Generating busperf %s report for %%{label}" % ctx.attr.format,
    )
    return [DefaultInfo(files = depset([out]), runfiles = ctx.runfiles([out]))]

busperf_analyze = rule(
    doc = "Runs `busperf analyze` to produce a text or HTML report.",
    implementation = _busperf_analyze_impl,
    attrs = {
        "vcd": attr.label(
            doc = "Simulation VCD.",
            allow_single_file = True,
            mandatory = True,
        ),
        "bus_yaml": attr.label(
            doc = "busperf YAML bus description.",
            allow_single_file = True,
            mandatory = True,
        ),
        "format": attr.string(doc = "Report format.", values = ["text", "html"], mandatory = True),
        "busperf_bin": attr.label(
            doc = "busperf binary.",
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
    },
)
