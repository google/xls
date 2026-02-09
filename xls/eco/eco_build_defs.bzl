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

"""
Build rules for generating and patching XLS IR to support ECO workflows.
"""

load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_opt_ir")
load("//xls/build_rules:xls_providers.bzl", "IrFileInfo")

EcopatchInfo = provider(
    doc = "Carries the generated patch file.",
    fields = {
        "patch": "File containing the IrPatchProto.",
        "before_ir": "Input opt IR file (before).",
        "after_ir": "Input opt IR file (after).",
    },
)

def _dslx_ir_diff_impl(ctx):
    before_ir = ctx.attr.before[IrFileInfo].ir_file
    after_ir = ctx.attr.after[IrFileInfo].ir_file

    # Ensure the ir2gxl tool's runfiles (its Python sources/deps) are hashed into
    # the action key so changes invalidate cached GXL outputs.
    ir2gxl_runfiles = ctx.attr.ir2gxl[DefaultInfo].default_runfiles.files.to_list()

    before_gxl = ctx.actions.declare_file(ctx.label.name + ".before.gxl")
    after_gxl = ctx.actions.declare_file(ctx.label.name + ".after.gxl")
    patch_out = ctx.actions.declare_file(ctx.label.name + ".patch.bin")

    # before.ir -> before.gxl
    ctx.actions.run_shell(
        inputs = [before_ir] + ctx.files.ir2gxl + ir2gxl_runfiles,
        outputs = [before_gxl],
        tools = [ctx.executable.ir2gxl],
        command = "bash -c '\"%s\" \"%s\" > \"%s\"'" %
                  (ctx.executable.ir2gxl.path, before_ir.path, before_gxl.path),
    )

    # after.ir -> after.gxl
    ctx.actions.run_shell(
        inputs = [after_ir] + ctx.files.ir2gxl + ir2gxl_runfiles,
        outputs = [after_gxl],
        tools = [ctx.executable.ir2gxl],
        mnemonic = "Ir2GxlAfter",
        command = "%s %s > %s" % (
            ctx.executable.ir2gxl.path,
            after_ir.path,
            after_gxl.path,
        ),
        use_default_shell_env = True,
    )

    timeout_flag = []
    if ctx.attr.timeout >= 0:
        timeout_flag = ["-t", str(ctx.attr.timeout)]

    verbosity_flag = []
    if ctx.attr.verbosity >= 0:
        verbosity_flag = ["-v", str(ctx.attr.verbosity)]

    mcs_flag = []
    if ctx.attr.mcs == False:
        mcs_flag = ["--no-mcs"]

    mcs_cutoff_flag = []
    if ctx.attr.mcs_cutoff >= 0:
        mcs_cutoff_flag = ["--mcs_cutoff=" + str(ctx.attr.mcs_cutoff)]

    ged_runfiles = ctx.attr.ged_main[DefaultInfo].default_runfiles.files.to_list()

    ctx.actions.run(
        inputs = [before_gxl, after_gxl] + ctx.files.ged_main + ged_runfiles,
        outputs = [patch_out],
        tools = [ctx.executable.ged_main],
        mnemonic = "GedMain",
        arguments = timeout_flag + verbosity_flag + mcs_flag + mcs_cutoff_flag + [
            "--before_ir=" + before_gxl.path,
            "--after_ir=" + after_gxl.path,
            "-p",
            patch_out.path,
        ],
        executable = ctx.executable.ged_main,
        use_default_shell_env = True,
    )

    return [
        # Match XLS style: expose the generated patch and source IRs.
        DefaultInfo(files = depset([patch_out, before_ir, after_ir])),
        OutputGroupInfo(
            patch = depset([patch_out]),
            opt_irs = depset([before_ir, after_ir]),
        ),
        EcopatchInfo(
            patch = patch_out,
            before_ir = before_ir,
            after_ir = after_ir,
        ),
    ]

xls_dslx_ir_diff_rule = rule(
    implementation = _dslx_ir_diff_impl,
    attrs = {
        "before": attr.label(
            providers = [IrFileInfo],
            doc = "Opt IR (before).",
        ),
        "after": attr.label(
            providers = [IrFileInfo],
            doc = "Opt IR (after).",
        ),
        "ir2gxl": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/eco:ir2gxl"),
        ),
        "ged_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/eco/mcs_ged:ged_main"),
        ),
        "mcs": attr.bool(
            default = True,
            doc = "Enable MCS when diffing; false passes --no-mcs to GED.",
        ),
        "timeout": attr.int(
            default = -1,
            doc = "Optional GED timeout in seconds; negative means omit flag.",
        ),
        "verbosity": attr.int(
            default = -1,
            doc = "Optional GED verbosity; negative means omit flag.",
        ),
        "mcs_cutoff": attr.int(
            default = -1,
            doc = "Stop MCS when remaining nodes <= this value; negative means run to completion.",
        ),
    },
)

def xls_dslx_ir_diff(name, srcs, dslx_top, timeout = None, mcs = None, verbosity = None, mcs_cutoff = None):
    """Builds opt IRs for two DSLX sources and emits a patch between them.

    Args:
      name: Base name for generated targets/outputs.
      srcs: List of two DSLX source labels [before, after].
      dslx_top: Either a single top name (applied to both) or a list of two.
    timeout: Optional GED timeout (seconds). None or negative => omit flag.
    mcs: Optional boolean. If false, pass --no-mcs to GED; true leaves default.
    verbosity: Optional GED verbosity. None or negative => omit flag.
    mcs_cutoff: Optional int. Stop MCS when remaining nodes <= this value; negative => run to completion.
    """
    if len(srcs) != 2:
        fail("xls_dslx_ir_diff.srcs must have length 2")
    if type(dslx_top) == type([]):
        if len(dslx_top) != 2:
            fail("xls_dslx_ir_diff.dslx_top list must have length 2")
        tops = dslx_top
    else:
        tops = [dslx_top, dslx_top]

    before_ir_target = name + "_before_opt_ir"
    after_ir_target = name + "_after_opt_ir"

    xls_dslx_opt_ir(
        name = before_ir_target,
        srcs = [srcs[0]],
        dslx_top = tops[0],
    )

    xls_dslx_opt_ir(
        name = after_ir_target,
        srcs = [srcs[1]],
        dslx_top = tops[1],
    )

    xls_dslx_ir_diff_rule_kwargs = dict(
        name = name,
        before = ":" + before_ir_target,
        after = ":" + after_ir_target,
    )
    if timeout != None:
        xls_dslx_ir_diff_rule_kwargs["timeout"] = timeout
    if mcs != None:
        xls_dslx_ir_diff_rule_kwargs["mcs"] = mcs
    if verbosity != None:
        xls_dslx_ir_diff_rule_kwargs["verbosity"] = verbosity
    if mcs_cutoff != None:
        xls_dslx_ir_diff_rule_kwargs["mcs_cutoff"] = mcs_cutoff

    xls_dslx_ir_diff_rule(**xls_dslx_ir_diff_rule_kwargs)

# Temporary aliases to ease migration.
xls_dslx_opt_ir_diff = xls_dslx_ir_diff
xls_dslx_eco = xls_dslx_ir_diff

def _xls_patch_ir_impl(ctx):
    diff_target = ctx.attr.ir_diff
    if not diff_target:
        fail("xls_patch_ir: 'ir_diff' label is required.")

    diff_info = diff_target[EcopatchInfo]
    ir_file = diff_info.before_ir
    patch_file = diff_info.patch
    after_ir = diff_info.after_ir

    output_ir = ctx.actions.declare_file(ctx.label.name + ".opt.ir")

    patch_ir_runfiles = ctx.attr.patch_ir_main[DefaultInfo].default_runfiles.files.to_list()

    inputs = [ir_file, patch_file] + ctx.files.patch_ir_main + patch_ir_runfiles
    args = [
        "--input_ir_path=" + ir_file.path,
        "--input_patch_path=" + patch_file.path,
        "--output_ir_path=" + output_ir.path,
    ]

    schedule_file = None
    if ctx.attr.schedule:
        schedule_files = ctx.files.schedule
        if len(schedule_files) != 1:
            fail("xls_patch_ir: expected exactly one file from 'schedule'")
        schedule_file = schedule_files[0]
        inputs.append(schedule_file)
        args.append("--input_schedule_path=" + schedule_file.path)

    ctx.actions.run(
        inputs = inputs,
        outputs = [output_ir],
        executable = ctx.executable.patch_ir_main,
        arguments = args,
        mnemonic = "PatchIr",
        use_default_shell_env = True,
    )

    check_inputs = [output_ir, after_ir]
    check_inputs += ctx.attr.check_ir_main[DefaultInfo].default_runfiles.files.to_list()
    check_inputs += ctx.files.check_ir_main
    check_flags = [
        "--match_exit_code=0",
        "--mismatch_exit_code=1",
    ]
    if ctx.attr.check_activation_count >= 0:
        check_flags.append("--activation_count=" + str(ctx.attr.check_activation_count))
    if ctx.attr.check_top:
        check_flags.append("--top=" + ctx.attr.check_top)
    if ctx.attr.check_timeout >= 0:
        check_flags.append("--timeout=" + str(ctx.attr.check_timeout) + "s")

    validation_stamp = ctx.actions.declare_file(ctx.label.name + ".equiv.ok")
    ctx.actions.run_shell(
        inputs = check_inputs,
        outputs = [validation_stamp],
        tools = [ctx.executable.check_ir_main],
        mnemonic = "CheckIrEquivalence",
        command = "set -euo pipefail\n\"%s\" \"$@\"\ntouch %s" % (
            ctx.executable.check_ir_main.path,
            validation_stamp.path,
        ),
        arguments = [output_ir.path, after_ir.path] + check_flags,
        use_default_shell_env = True,
    )

    return [
        DefaultInfo(files = depset([output_ir, validation_stamp])),
        IrFileInfo(ir_file = output_ir),
    ]

xls_patch_ir_rule = rule(
    implementation = _xls_patch_ir_impl,
    attrs = {
        "ir_diff": attr.label(
            providers = [EcopatchInfo],
            doc = "xls_dslx_ir_diff target providing the before IR and patch.",
        ),
        "schedule": attr.label(
            allow_single_file = True,
            doc = "Optional PackageScheduleProto textproto to patch alongside IR.",
        ),
        "patch_ir_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/eco:patch_ir_main"),
            doc = "Binary used to apply the patch.",
        ),
        "check_ir_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/dev_tools:check_ir_equivalence_main"),
            doc = "Binary used to validate equivalence of patched IR vs. revised IR.",
        ),
        "check_top": attr.string(
            default = "",
            doc = "Optional top entity passed to check_ir_equivalence_main.",
        ),
        "check_activation_count": attr.int(
            default = -1,
            doc = "Optional activation count for proc equivalence; negative omits the flag.",
        ),
        "check_timeout": attr.int(
            default = -1,
            doc = "Optional timeout (seconds) for check_ir_equivalence_main; negative omits flag.",
        ),
    },
)

def xls_patch_ir(
        name,
        ir_diff,
        schedule = None,
        top = None,
        activation_count = None,
        timeout = None,
        **kwargs):
    """Applies an ECO patch to the before IR and validates against the after IR."""
    if "check_activation_count" in kwargs and activation_count != None:
        fail("Provide only one of activation_count or check_activation_count.")
    if "check_top" in kwargs and top != None:
        fail("Provide only one of top or check_top.")
    if "check_timeout" in kwargs and timeout != None:
        fail("Provide only one of timeout or check_timeout.")

    rule_kwargs = dict(kwargs)
    rule_kwargs["ir_diff"] = ir_diff
    if schedule != None:
        rule_kwargs["schedule"] = schedule
    if activation_count != None:
        rule_kwargs["check_activation_count"] = activation_count
    if top != None:
        rule_kwargs["check_top"] = top
    if timeout != None:
        rule_kwargs["check_timeout"] = timeout

    xls_patch_ir_rule(
        name = name,
        **rule_kwargs
    )
