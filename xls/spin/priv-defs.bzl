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

"""Internal Bazel rules for xls/spin -- not part of the public API.

Used by testdata/BUILD.
For the public-facing rules see //xls/spin:defs.bzl.

Rules:
  spin_syntax_check -- validate Promela syntax at build time              (build action)
  spin_trace        -- run SPIN as a build action, write JSON artifact    (build action)
  spin_trace_test   -- compare SPIN and DSLX traces per channel           (bazel test)
  dslx_trace    -- generate a DSLX channel-event trace textproto     (build action)

Macros:
  spin_ir_golden_test -- compile DSLX to IR and diff against a checked-in golden
                         (builds {name}_gen_ir; tests with {name}; updates with
                         {name}_update_golden)
"""

load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "ACTION_NAMES",
)
load(
    "@bazel_tools//tools/cpp:toolchain_utils.bzl",
    "find_cpp_toolchain",
    "use_cpp_toolchain",
)
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_ir",
)
load("//xls/build_rules:xls_diff_test.bzl", "diff_test")

def _spin_syntax_check_impl(ctx):
    stamp = ctx.actions.declare_file(ctx.label.name + ".syntax_ok")

    # spin -a hardcodes its preprocessor; -P overrides it at runtime so we
    # pass clang in preprocessor-only mode instead of a system gcc.
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    c_compiler = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.c_compile,
    )

    ctx.actions.run_shell(
        inputs = depset(
            [ctx.file.src, ctx.executable._spin],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [stamp],
        command = "{spin} -a -P\"{compiler} -E -x c\" {pml} && touch {stamp}".format(
            spin = ctx.executable._spin.path,
            compiler = c_compiler,
            pml = ctx.file.src.path,
            stamp = stamp.path,
        ),
        mnemonic = "PromelaSyntax",
        progress_message = "Checking Promela syntax of %{input}",
    )
    return [DefaultInfo(files = depset([stamp]))]

spin_syntax_check = rule(
    implementation = _spin_syntax_check_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".pml"],
            mandatory = True,
            doc = "Promela source file to validate.",
        ),
        "_spin": attr.label(
            default = Label("@spin//:spin"),
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
    doc = """\
Validates Promela syntax at build time (spin -a).

The build fails with SPIN's diagnostics if the model is syntactically invalid.
Unlike a test, this is enforced on every `bazel build`.
""",
)

def _spin_trace_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".json")
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    c_compiler = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.c_compile,
    )
    ctx.actions.run_shell(
        inputs = depset(
            [ctx.file.src, ctx.executable._spin],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [out],
        command = "{spin} -Q {out} -P\"{compiler} -E -x c\" {args} {src}".format(
            spin = ctx.executable._spin.path,
            out = out.path,
            compiler = c_compiler,
            args = " ".join(ctx.attr.spin_args),
            src = ctx.file.src.path,
        ),
        mnemonic = "PromelaSimTrace",
        progress_message = "SPIN simulation trace of %{input}",
    )
    return [DefaultInfo(files = depset([out]))]

spin_trace = rule(
    implementation = _spin_trace_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".pml"],
            mandatory = True,
            doc = "Promela source file.",
        ),
        "spin_args": attr.string_list(
            default = ["-c"],
            doc = "Flags forwarded to spin before the model path (default: -c).",
        ),
        "_spin": attr.label(
            default = Label("@spin//:spin"),
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
    doc = """\
Runs SPIN as a build action and writes the channel-event JSON artifact.

  spin_trace(
      name = "my_trace",
      src  = ":my_pml",
  )

  bazel build //path/to:my_trace   # produces my_trace.json in bazel-bin/
""",
)

def _spin_trace_test_impl(ctx):
    spin_json = ctx.attr.spin_trace[DefaultInfo].files.to_list()[0]
    dslx_json = ctx.attr.dslx_trace[DefaultInfo].files.to_list()[0]
    compare = ctx.executable._compare
    dslx_src = ctx.file.dslx_file

    cmd = "{compare} --spin_trace={spin} --dslx_trace={dslx} --dslx_file={src}".format(
        compare = compare.short_path,
        spin = spin_json.short_path,
        dslx = dslx_json.short_path,
        src = dslx_src.short_path,
    )
    extra_files = [compare, spin_json, dslx_json, dslx_src]

    script = ctx.actions.declare_file(ctx.label.name + "_compare_test.sh")
    ctx.actions.write(
        output = script,
        is_executable = True,
        content = "\n".join([
            "#!/usr/bin/env bash",
            "set -e",
            cmd,
        ]),
    )
    runfiles = ctx.runfiles(files = extra_files)
    runfiles = runfiles.merge(ctx.attr._compare[DefaultInfo].default_runfiles)
    return [DefaultInfo(executable = script, runfiles = runfiles)]

spin_trace_test = rule(
    implementation = _spin_trace_test_impl,
    test = True,
    attrs = {
        "spin_trace": attr.label(
            mandatory = True,
            doc = "Target producing the SPIN trace JSON (spin_trace).",
        ),
        "dslx_trace": attr.label(
            mandatory = True,
            doc = "Target producing the DSLX trace JSON (dslx_trace).",
        ),
        "dslx_file": attr.label(
            mandatory = True,
            allow_single_file = [".x"],
            doc = "DSLX source file. Derives channel-name rewriting and " +
                  "proc-hierarchy paths from source.",
        ),
        "_compare": attr.label(
            default = Label("//xls/spin:promela_trace_compare"),
            executable = True,
            cfg = "target",
        ),
    },
    doc = """\
Compares SPIN and DSLX channel-event traces per channel as a Bazel test.

The test passes when every channel's event sequence (values and direction)
is identical in both traces.  Events on different channels may interleave
differently; only the relative order within each channel is checked.

  spin_trace_test(
      name = "my_trace_compare",
      spin_trace = ":my_spin_trace",
      dslx_trace = ":my_dslx_trace",
      dslx_file = ":my_proc.x",
  )

  bazel test //path/to:my_trace_compare
""",
)

def _dslx_trace_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".textproto")
    src = ctx.file.src
    args = [
        src.path,
        "--trace_channels",
        "--output_results_proto=" + out.path,
        "--dslx_path=" + src.dirname,
    ]
    for p in ctx.attr.dslx_paths:
        args.append("--dslx_path=" + p)
    ctx.actions.run(
        executable = ctx.executable._interpreter,
        arguments = args,
        inputs = [src],
        outputs = [out],
        mnemonic = "DslxTrace",
        progress_message = "Generating DSLX channel-event trace from %{input}",
    )
    return [DefaultInfo(files = depset([out]))]

dslx_trace = rule(
    implementation = _dslx_trace_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".x"],
            mandatory = True,
            doc = "DSLX source file to interpret.",
        ),
        "dslx_paths": attr.string_list(
            default = [],
            doc = "Additional --dslx_path entries for resolving imports.",
        ),
        "_interpreter": attr.label(
            default = Label("//xls/dslx:interpreter_main"),
            executable = True,
            cfg = "exec",
        ),
    },
    doc = """\
Generates a DSLX channel-event trace as a build artifact.

Runs the DSLX interpreter with --trace_channels --output_results_proto and
writes the EvaluatorResultsProto textproto to <name>.textproto.  Suitable
as the dslx_trace input to spin_trace_test.
""",
)

def spin_ir_golden_test(name, src, golden, dslx_top, convert_tests = False, tags = []):
    """Generates IR from a DSLX source and diffs it against a golden .ir file.

    Creates two targets:
      {name}_gen_ir  -- xls_dslx_ir build action; produces the .ir artifact.
      {name}         -- diff_test; fails if the generated IR differs from the
                        golden.  A companion {name}_update_golden target is
                        created automatically for refreshing the golden locally.

    Typical usage in testdata/BUILD:

      spin_ir_golden_test(
          name = "counter_ir_golden",
          src  = "counter.x",
          golden = "counter.ir",
          dslx_top = "CounterTest",
          convert_tests = True,
      )

    Args:
      name:          Base name prefix for generated targets.
      src:           DSLX source file (e.g. "counter.x").
      golden:        Checked-in golden IR file (e.g. "counter.ir").
      dslx_top:      Top-level proc name passed to ir_converter_main.
      convert_tests: Pass --convert_tests to ir_converter_main so #[test_proc]
                     procs are included in the converted IR.  Required when
                     dslx_top names a test proc.
      tags:          Extra tags forwarded to the diff_test target.
    """
    ir_conv_args = {"lower_to_proc_scoped_channels": "true"}
    if convert_tests:
        ir_conv_args["convert_tests"] = "true"

    xls_dslx_ir(
        name = name + "_gen_ir",
        srcs = [src],
        dslx_top = dslx_top,
        ir_conv_args = ir_conv_args,
    )

    diff_test(
        name = name,
        file = ":" + name + "_gen_ir.ir",
        golden = golden,
        tags = tags,
    )
