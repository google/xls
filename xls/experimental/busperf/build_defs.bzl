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

"""Public API for generating a busperf YAML bus description for a DSLX proc."""

visibility(["//xls/experimental/busperf/..."])

def _busperf_yaml_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".busperf.yaml")

    args = ctx.actions.args()
    args.add(ctx.file.signature)
    args.add("--scope", ctx.attr.scope)
    args.add("--output", out)

    ctx.actions.run(
        executable = ctx.executable.xls_sig_to_busperf,
        arguments = [args],
        inputs = [ctx.file.signature],
        outputs = [out],
        mnemonic = "BusperfYaml",
        progress_message = "Generating busperf YAML for %{label}",
    )
    return [DefaultInfo(files = depset([out]), runfiles = ctx.runfiles([out]))]

busperf_yaml = rule(
    doc = ("Runs xls_sig_to_busperf to produce a busperf YAML bus " +
           "description from an existing codegen signature, e.g. the " +
           "`<name>.sig.textproto` output of an xls_dslx_verilog target."),
    implementation = _busperf_yaml_impl,
    attrs = {
        "signature": attr.label(
            doc = "Top block's ModuleSignatureProto.",
            allow_single_file = True,
            mandatory = True,
        ),
        "scope": attr.string(
            doc = "Dot-separated VCD scope path to the DUT, e.g. \"tb_foo.dut\".",
            mandatory = True,
        ),
        "xls_sig_to_busperf": attr.label(
            doc = "xls_sig_to_busperf binary.",
            executable = True,
            cfg = "exec",
            default = "//xls/experimental/busperf:xls_sig_to_busperf",
        ),
    },
)
