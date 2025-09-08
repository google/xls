# Copyright 2021 The XLS Authors
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
This module contains codegen-related build rules for XLS.
"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "append_default_to_args",
    "args_to_string",
    "get_input_infos",
    "get_output_filename_value",
    "get_runfiles_for_xls",
    "get_src_ir_for_xls",
    "get_transitive_built_files_for_xls",
    "is_args_valid",
    "split_filename",
)
load("//xls/build_rules:xls_config_rules.bzl", "CONFIG")
load("//xls/build_rules:xls_ir_rules.bzl", "xls_ir_common_attrs")
load(
    "//xls/build_rules:xls_providers.bzl",
    "CODEGEN_FIELDS",
    "CodegenInfo",
    "ConvIrInfo",
    "SCHEDULING_FIELDS",
)
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "xls_toolchain_attrs",
)

_DEFAULT_SCHEDULING_ARGS = {
    "delay_model": "unit",
}

_DEFAULT_CODEGEN_ARGS = {
    "register_merge_strategy": "IdentityOnly",
    "emit_sv_types": True,
}

_SYSTEM_VERILOG_FILE_EXTENSION = "sv"
_VERILOG_FILE_EXTENSION = "v"
_SIGNATURE_TEXTPROTO_FILE_EXTENSION = ".sig.textproto"
_SCHEDULE_TEXTPROTO_FILE_EXTENSION = ".schedule.textproto"
_VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION = ".verilog_line_map.textproto"
_BLOCK_METRICS_FILE_EXTENSION = ".metrics.textproto"
_BLOCK_IR_FILE_EXTENSION = ".block.ir"
_SCHEDULE_IR_FILE_EXTENSION = ".schedule.opt.ir"
_CODEGEN_LOG_FILE_EXTENSION = ".codegen.log"
_CODEGEN_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION = ".codegen_options.textproto"
_SCHEDULING_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION = ".schedule_options.textproto"
_CODEGEN_FLAGS = CODEGEN_FIELDS.keys()
_SCHEDULING_FLAGS = SCHEDULING_FIELDS.keys()

xls_ir_verilog_attrs = {
    "codegen_args": attr.string_dict(
        doc = "Arguments of the codegen tool. For details on the arguments, " +
              "refer to the codegen_main application at " +
              "//xls/tools/codegen_main.cc.",
    ),
    "codegen_options_proto": attr.label(
        allow_single_file = True,
        default = None,
        doc = "Filename of a protobuf with arguments of the codegen tool. " +
              "For details on the arguments, " +
              "refer to the codegen_main application at " +
              "//xls/tools/codegen_main.cc.",
    ),
    "scheduling_options_proto": attr.label(
        allow_single_file = True,
        default = None,
        doc = "Filename of a protobuf with scheduling options arguments " +
              "of the codegen tool. For details on the arguments, " +
              "refer to the codegen_main application at " +
              "//xls/tools/codegen_main.cc.",
    ),
    "verilog_file": attr.output(
        doc = "The filename of Verilog file generated. The filename must " +
              "have a " + _VERILOG_FILE_EXTENSION + " extension.",
        mandatory = True,
    ),
    "module_sig_file": attr.output(
        doc = "The filename of module signature of the generated Verilog " +
              "file. If not specified, the basename of the Verilog file " +
              "followed by a " + _SIGNATURE_TEXTPROTO_FILE_EXTENSION + " " +
              "extension is used.",
    ),
    "schedule_file": attr.output(
        doc = "The filename of schedule of the generated Verilog file." +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _SCHEDULE_TEXTPROTO_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "verilog_line_map_file": attr.output(
        doc = "The filename of line map for the generated Verilog file." +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "schedule_ir_file": attr.output(
        doc = "The filename of scheduled IR file generated during scheduled. " +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _SCHEDULE_IR_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "block_ir_file": attr.output(
        doc = "The filename of block-level IR file generated during codegen. " +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _BLOCK_IR_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "codegen_log_file": attr.output(
        doc = "The filename to log stderr to. " +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _CODEGEN_LOG_FILE_EXTENSION + " extension is used.",
    ),
    "codegen_options_used_textproto_file": attr.output(
        doc = "The filename to write the full configuration options used for " +
              "this codegen. If not specified, the basename of the Verilog " +
              "file followed by a " +
              _CODEGEN_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION + " extension " +
              "is used.",
    ),
    "scheduling_options_used_textproto_file": attr.output(
        doc = "The filename to write the full configuration options used for " +
              "this scheduling. If not specified, the basename of the Verilog " +
              "file followed by a " +
              _SCHEDULING_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION + " extension " +
              "is used.",
    ),
    "block_metrics_file": attr.output(
        doc = "The filename to write the metrics, including the bill of materials, " +
              "for the generated Verilog file. If not specified, the basename of the " +
              "Verilog file followed by a " +
              _BLOCK_METRICS_FILE_EXTENSION + " extension is used.",
    ),
}

def _is_combinational_generator(arguments):
    """Returns True, if "generator" is "combinational". Otherwise, returns False.

    Args:
      arguments: The list of arguments.
    Returns:
      Returns True, if "generator" is "combinational". Otherwise, returns False.
    """
    return arguments.get("generator", "") == "combinational"

def _uses_fdo(arguments):
    """Returns True, if codegen options specify using FDO. Otherwise, returns False.

    Args:
      arguments: The list of arguments.
    Returns:
      Returns True, if codegen will use FDO. Otherwise, returns False.
    """
    return (arguments.get("use_fdo", "false").lower() == "true")

def append_xls_ir_verilog_generated_files(args, basename, arguments):
    """Returns a dictionary of arguments appended with filenames generated by the 'xls_ir_verilog' rule.

    Args:
      args: A dictionary of arguments.
      basename: The file basename.
      arguments: The codegen arguments.

    Returns:
      Returns a dictionary of arguments appended with filenames generated by the 'xls_ir_verilog' rule.
    """
    args.setdefault(
        "module_sig_file",
        basename + _SIGNATURE_TEXTPROTO_FILE_EXTENSION,
    )
    args.setdefault(
        "schedule_ir_file",
        basename + _SCHEDULE_IR_FILE_EXTENSION,
    )
    args.setdefault(
        "block_ir_file",
        basename + _BLOCK_IR_FILE_EXTENSION,
    )
    if not _is_combinational_generator(arguments):
        args.setdefault(
            "schedule_file",
            basename + _SCHEDULE_TEXTPROTO_FILE_EXTENSION,
        )
    args.setdefault(
        "verilog_line_map_file",
        basename + _VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION,
    )
    args.setdefault(
        "codegen_log_file",
        basename + _CODEGEN_LOG_FILE_EXTENSION,
    )
    args.setdefault(
        "codegen_options_used_textproto_file",
        basename + _CODEGEN_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION,
    )
    args.setdefault(
        "scheduling_options_used_textproto_file",
        basename + _SCHEDULING_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION,
    )
    args.setdefault(
        "block_metrics_file",
        basename + _BLOCK_METRICS_FILE_EXTENSION,
    )
    return args

def get_xls_ir_verilog_generated_files(args, arguments):
    """Returns a list of filenames generated by the 'xls_ir_verilog' rule found in 'args'.

    Args:
      args: A dictionary of arguments.
      arguments: The codegen arguments.

    Returns:
      Returns a list of filenames generated by the 'xls_ir_verilog' rule found in 'args'.
    """
    generated_files = [
        args.get("module_sig_file"),
        args.get("schedule_ir_file"),
        args.get("block_ir_file"),
        args.get("verilog_line_map_file"),
    ]
    if not _is_combinational_generator(arguments):
        generated_files.append(args.get("schedule_file"))
    return generated_files

def validate_verilog_filename(verilog_filename, use_system_verilog):
    """Validate verilog filename.

    Args:
      verilog_filename: The verilog filename to validate.
      use_system_verilog: Whether to validate the file name as system verilog or not.

    Produces a failure if the verilog filename does not have a basename or a
    valid extension.
    """

    if (use_system_verilog and
        split_filename(verilog_filename)[-1] != _SYSTEM_VERILOG_FILE_EXTENSION):
        fail("SystemVerilog filename must contain the '%s' extension." %
             _SYSTEM_VERILOG_FILE_EXTENSION)

    if (not use_system_verilog and
        split_filename(verilog_filename)[-1] != _VERILOG_FILE_EXTENSION):
        fail("Verilog filename must contain the '%s' extension." %
             _VERILOG_FILE_EXTENSION)

def xls_ir_verilog_impl(ctx, src, conv_info):
    """The core implementation of the 'xls_ir_verilog' rule.

    Generates a Verilog file, module signature file, block file, Verilog line
    map, and schedule file.

    Args:
      ctx: The current rule's context object.
      src: The source file.
      conv_info: The ConvIrInfo for the source containing original input files and any interface proto.

    Returns:
      A tuple with the following elements in the order presented:
        1. The CodegenInfo provider
        1. The list of built files.
        1. The runfiles.
    """
    codegen_tool = ctx.executable._xls_codegen_tool
    my_generated_files = []
    runfiles_list = [src.ir_file] + conv_info.original_input_files

    # default arguments
    if ctx.file.codegen_options_proto == None:
        codegen_args = append_default_to_args(
            ctx.attr.codegen_args,
            _DEFAULT_CODEGEN_ARGS,
        )
    else:
        codegen_args = ctx.attr.codegen_args

    if ctx.file.scheduling_options_proto == None:
        codegen_args = append_default_to_args(
            codegen_args,
            _DEFAULT_SCHEDULING_ARGS,
        )

    # parse arguments

    is_args_valid(codegen_args, _CODEGEN_FLAGS + _SCHEDULING_FLAGS)
    codegen_str_args = args_to_string(codegen_args, _CODEGEN_FLAGS + _SCHEDULING_FLAGS)
    uses_combinational_generator = _is_combinational_generator(codegen_args)
    final_args = codegen_str_args
    if (ctx.file.codegen_options_proto == None and conv_info.ir_interface != None):
        # Mixing proto options and normal ones is not supported. If proto
        # options are being used the user will need to have put the interface
        # proto in manually.
        final_args += " --ir_interface_proto={}".format(conv_info.ir_interface.path)
        runfiles_list.append(conv_info.ir_interface)

    uses_fdo = _uses_fdo(codegen_args)
    if (uses_fdo):
        fail("\n\n*** To enable FDO, use the 'xls_ir_verilog_fdo' rule instead.\n\n")

    # output filenames
    verilog_filename = ctx.attr.verilog_file.name
    if codegen_args.get("use_system_verilog", "") == "":
        use_system_verilog = split_filename(verilog_filename)[-1] != "v"
    else:
        use_system_verilog = codegen_args.get("use_system_verilog", "True").lower() == "true"
    validate_verilog_filename(verilog_filename, use_system_verilog)
    verilog_basename = split_filename(verilog_filename)[0]

    verilog_line_map_filename = get_output_filename_value(
        ctx,
        "verilog_line_map_file",
        verilog_basename + _VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION,
    )
    verilog_line_map_file = ctx.actions.declare_file(verilog_line_map_filename)
    my_generated_files.append(verilog_line_map_file)
    final_args += " --output_verilog_line_map_path={}".format(verilog_line_map_file.path)

    schedule_file = None
    if not uses_combinational_generator:
        # Pipeline generator produces a schedule artifact.
        schedule_filename = get_output_filename_value(
            ctx,
            "schedule_file",
            verilog_basename + _SCHEDULE_TEXTPROTO_FILE_EXTENSION,
        )

        schedule_file = ctx.actions.declare_file(schedule_filename)
        my_generated_files.append(schedule_file)
        final_args += " --output_schedule_path={}".format(schedule_file.path)

    verilog_file = ctx.actions.declare_file(verilog_filename)
    module_sig_filename = get_output_filename_value(
        ctx,
        "module_sig_file",
        verilog_basename + _SIGNATURE_TEXTPROTO_FILE_EXTENSION,
    )
    module_sig_file = ctx.actions.declare_file(module_sig_filename)
    my_generated_files += [verilog_file, module_sig_file]
    final_args += " --output_verilog_path={}".format(verilog_file.path)
    final_args += " --output_signature_path={}".format(module_sig_file.path)
    schedule_ir_filename = get_output_filename_value(
        ctx,
        "schedule_ir_file",
        verilog_basename + _SCHEDULE_IR_FILE_EXTENSION,
    )
    schedule_ir_file = ctx.actions.declare_file(schedule_ir_filename)
    final_args += " --output_schedule_ir_path={}".format(schedule_ir_file.path)
    my_generated_files.append(schedule_ir_file)
    block_ir_filename = get_output_filename_value(
        ctx,
        "block_ir_file",
        verilog_basename + _BLOCK_IR_FILE_EXTENSION,
    )
    block_ir_file = ctx.actions.declare_file(block_ir_filename)
    final_args += " --output_block_ir_path={}".format(block_ir_file.path)
    my_generated_files.append(block_ir_file)

    # Get runfiles
    codegen_tool_runfiles = ctx.attr._xls_codegen_tool[DefaultInfo].default_runfiles

    if ctx.file.codegen_options_proto:
        final_args += " --codegen_options_proto={}".format(ctx.file.codegen_options_proto.path)
        runfiles_list.append(ctx.file.codegen_options_proto)

    if ctx.file.scheduling_options_proto:
        final_args += " --scheduling_options_proto={}".format(ctx.file.scheduling_options_proto.path)
        runfiles_list.append(ctx.file.scheduling_options_proto)

    runfiles = get_runfiles_for_xls(ctx, [codegen_tool_runfiles], runfiles_list)

    tools = [codegen_tool]

    codegen_log_filename = get_output_filename_value(
        ctx,
        "codegen_log_file",
        verilog_basename + _CODEGEN_LOG_FILE_EXTENSION,
    )
    log_file = ctx.actions.declare_file(codegen_log_filename)
    my_generated_files.append(log_file)

    block_metrics_filename = get_output_filename_value(
        ctx,
        "block_metrics_file",
        verilog_basename + _BLOCK_METRICS_FILE_EXTENSION,
    )
    block_metrics_file = ctx.actions.declare_file(block_metrics_filename)
    my_generated_files.append(block_metrics_file)
    final_args += " --block_metrics_path={}".format(
        block_metrics_file.path,
    )

    codegen_config_textproto_file = get_output_filename_value(
        ctx,
        "codegen_options_used_textproto_file",
        verilog_basename + _CODEGEN_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION,
    )
    config_textproto_file = ctx.actions.declare_file(codegen_config_textproto_file)
    my_generated_files.append(config_textproto_file)
    final_args += " --codegen_options_used_textproto_file={}".format(
        config_textproto_file.path,
    )
    schedule_config_textproto_file = get_output_filename_value(
        ctx,
        "scheduling_options_used_textproto_file",
        verilog_basename + _SCHEDULING_OPTIONS_USED_TEXTPROTO_FILE_EXTENSION,
    )
    sched_config_textproto_file = ctx.actions.declare_file(schedule_config_textproto_file)
    my_generated_files.append(sched_config_textproto_file)
    final_args += " --scheduling_options_used_textproto_file={}".format(
        sched_config_textproto_file.path,
    )

    if "local" in ctx.attr.tags:
        execution_requirements = {"no-remote-exec": "1"}
    else:
        execution_requirements = None

    ctx.actions.run_shell(
        outputs = my_generated_files,
        tools = tools,
        inputs = runfiles.files,
        execution_requirements = execution_requirements,
        command = "set -o pipefail; {} {} {} 2>&1 | tee {}".format(
            codegen_tool.path,
            src.ir_file.path,
            final_args,
            log_file.path,
        ),
        mnemonic = "GenerateVerilog",
        progress_message = "Compiling %s" % verilog_file.short_path,
        toolchain = None,
    )

    # Set top to match module_name if it is set
    if "module_name" in codegen_args:
        codegen_args = {k: v for k, v in codegen_args.items()}
        codegen_args["top"] = codegen_args["module_name"]
    return [
        CodegenInfo(
            input_ir = src,
            verilog_file = verilog_file,
            module_sig_file = module_sig_file,
            verilog_line_map_file = verilog_line_map_file,
            schedule_file = schedule_file,
            schedule_ir_file = schedule_ir_file,
            block_ir_file = block_ir_file,
            **codegen_args
        ),
        my_generated_files,
        runfiles,
    ]

def _xls_ir_verilog_impl_wrapper(ctx):
    """The implementation of the 'xls_ir_verilog' rule.

    Wrapper for xls_ir_verilog_impl. See: xls_ir_verilog_impl.

    Args:
      ctx: The current rule's context object.
    Returns:
      CodegenInfo provider
      DefaultInfo provider
    """
    if ConvIrInfo in ctx.attr.src:
        conv_info = ctx.attr.src[ConvIrInfo]
    else:
        conv_info = ConvIrInfo(original_input_files = ctx.files.src, ir_interface = None)

    codegen_info, built_files_list, runfiles = xls_ir_verilog_impl(
        ctx,
        get_src_ir_for_xls(ctx),
        conv_info,
    )

    return [
        codegen_info,
        DefaultInfo(
            files = depset(
                direct = built_files_list,
                transitive = get_transitive_built_files_for_xls(
                    ctx,
                    [ctx.attr.src],
                ),
            ),
            runfiles = runfiles,
        ),
    ] + get_input_infos(ctx.attr.src)

xls_ir_verilog = rule(
    doc = """A build rule that generates a Verilog file from an IR file.

Example:

    ```
    xls_ir_verilog(
        name = "a_verilog",
        src = "a.ir",
        codegen_args = {
            "pipeline_stages": "1",
            ...
        },
    )
    ```
    """,
    implementation = _xls_ir_verilog_impl_wrapper,
    attrs = dicts.add(
        xls_ir_common_attrs,
        xls_ir_verilog_attrs,
        CONFIG["xls_outs_attrs"],
        dicts.pick(xls_toolchain_attrs, ["_xls_codegen_tool"]),
        {
            "log_file": attr.label(
                doc = "The file to log the output to.",
                mandatory = False,
                allow_single_file = True,
            ),
        },
    ),
)

def _xls_benchmark_verilog_impl(ctx):
    """Implementation of the 'xls_benchmark_verilog' rule.

    Computes and prints various metrics about a Verilog target.

    Args:
      ctx: The current rule's context object.
    Returns:
      DefaultInfo provider
    """
    benchmark_codegen_tool = ctx.executable._xls_benchmark_codegen_tool
    codegen_info = ctx.attr.verilog_target[CodegenInfo]
    opt_ir_info = codegen_info.input_ir
    if not hasattr(codegen_info, "top"):
        fail("Verilog target '%s' does not provide a top value" %
             ctx.attr.verilog_target.label.name)
    cmd = "{tool} {opt_ir} {block_ir} {verilog} --top={top}".format(
        opt_ir = opt_ir_info.ir_file.short_path,
        verilog = codegen_info.verilog_file.short_path,
        top = codegen_info.top,
        tool = benchmark_codegen_tool.short_path,
        block_ir = codegen_info.block_ir_file.short_path,
    )
    for flag in _CODEGEN_FLAGS + _SCHEDULING_FLAGS:
        if flag in ["input_ir", "top", "verilog_file", "block_ir_file"]:
            # already handled above
            continue
        value = getattr(codegen_info, flag, None)
        if value != None:
            cmd += " --{flag}={value}".format(flag = flag, value = value)
    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")

    # Get runfiles
    benchmark_codegen_tool_runfiles = (
        ctx.attr._xls_benchmark_codegen_tool[DefaultInfo].default_runfiles
    )
    runfiles = get_runfiles_for_xls(
        ctx,
        [benchmark_codegen_tool_runfiles],
        [
            opt_ir_info.ir_file,
            codegen_info.block_ir_file,
            codegen_info.verilog_file,
        ],
    )

    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/usr/bin/env bash",
            "set -e",
            cmd,
            "exit 0",
        ]),
        is_executable = True,
    )
    return [
        DefaultInfo(
            runfiles = runfiles,
            files = depset(
                direct = [executable_file],
                transitive = get_transitive_built_files_for_xls(
                    ctx,
                    [ctx.attr.verilog_target],
                ),
            ),
            executable = executable_file,
        ),
    ] + get_input_infos(ctx.attr.verilog_target)

xls_benchmark_verilog_attrs = {
    "verilog_target": attr.label(
        doc = "The verilog target to benchmark.",
        providers = [CodegenInfo],
    ),
}

xls_benchmark_verilog = rule(
    doc = """Computes and prints various metrics about a Verilog target.

Example:
    ```
    xls_benchmark_verilog(
        name = "a_benchmark",
        verilog_target = "a_verilog_target",
    )
    ```
    """,
    implementation = _xls_benchmark_verilog_impl,
    attrs = dicts.add(
        xls_benchmark_verilog_attrs,
        dicts.pick(xls_toolchain_attrs, ["_xls_benchmark_codegen_tool"]),
    ),
    executable = True,
)
