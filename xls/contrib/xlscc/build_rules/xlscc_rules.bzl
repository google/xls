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

"""Build rules to compile with xlscc"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "append_xls_ir_verilog_generated_files",
    "get_xls_ir_verilog_generated_files",
    "xls_ir_verilog_attrs",
    "xls_ir_verilog_impl",
)
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "append_default_to_args",
    "args_to_string",
    "get_output_filename_value",
    "is_args_valid",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "CONFIG",
    "enable_generated_file_wrapper",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "append_xls_ir_opt_ir_generated_files",
    "get_xls_ir_opt_ir_generated_files",
    "xls_ir_opt_ir_attrs",
    "xls_ir_opt_ir_impl",
)
load(
    "//xls/build_rules:xls_providers.bzl",
    "ConvIrInfo",
    "IrFileInfo",
)
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "xls_toolchain_attrs",
)
load(
    "//xls/build_rules:xls_type_check_utils.bzl",
    "bool_type_check",
    "dictionary_type_check",
    "list_type_check",
    "string_type_check",
)
load(
    "//xls/contrib/xlscc/build_rules:xlscc_providers.bzl",
    "XlsccIncludeInfo",
    "XlsccInfo",
)

_CC_FILE_EXTENSION = ".cc"
_H_FILE_EXTENSION = ".h"
_INC_FILE_EXTENSION = ".inc"
_IR_FILE_EXTENSION = ".ir"
_PROTOBIN_FILE_EXTENSION = ".protobin"
_BINARYPB_FILE_EXTENSION = ".binarypb"
_PROTOTEXT_FILE_EXTENSION = ".pbtxt"
_TEXTPROTO_FILE_EXTENSION = ".txtpb"
_DEFAULT_XLSCC_ARGS = {
    "top": "Run",
}

def _append_xls_cc_ir_generated_files(args, basename):
    """Returns a dictionary of arguments appended with filenames generated by the 'xls_cc_ir' rule.

    Args:
      args: A dictionary of arguments.
      basename: The file basename.

    Returns:
      Returns a dictionary of arguments appended with filenames generated by the 'xls_cc_ir' rule.
    """
    args.setdefault("ir_file", basename + _IR_FILE_EXTENSION)
    return args

def _get_xls_cc_ir_generated_files(args):
    """Returns a list of filenames generated by the 'xls_cc_ir' rule found in 'args'.

    Args:
      args: A dictionary of arguments.

    Returns:
      Returns a list of files generated by the 'xls_cc_ir' rule found in 'args'.
    """
    return [args.get("ir_file")]

def _get_xls_cc_ir_source_files(ctx):
    files = ([ctx.file.src] +
             ctx.files._default_cc_header_files +
             ctx.files._default_synthesis_header_files +
             ctx.files.src_deps)
    if ctx.file.block:
        files.append(ctx.file.block)
    return files

def _get_runfiles_for_xls_cc_ir(ctx):
    """Returns the runfiles from a 'xls_cc_ir' ctx.

    Args:
      ctx: The current rule's context object.

    Returns:
      The runfiles from a 'xls_cc_ir' ctx.
    """
    transitive_runfiles = []

    files = _get_xls_cc_ir_source_files(ctx)
    runfiles = ctx.runfiles(files = files)
    transitive_runfiles.append(ctx.attr
        ._xlscc_tool[DefaultInfo].default_runfiles)
    transitive_runfiles.append(ctx.attr
        ._default_cc_header_files[DefaultInfo].default_runfiles)
    transitive_runfiles.append(ctx.attr
        ._default_synthesis_header_files[DefaultInfo].default_runfiles)
    for dep in ctx.attr.src_deps:
        transitive_runfiles.append(dep[DefaultInfo].default_runfiles)

    runfiles = runfiles.merge_all(transitive_runfiles)
    return runfiles

def _get_headers_for_xls_cc_ir(ctx):
    transitive_runfiles = []
    files = (ctx.files._default_cc_header_files +
             ctx.files._default_synthesis_header_files +
             ctx.files.src_deps)
    runfiles = ctx.runfiles(files = files)
    for dep in ctx.attr.src_deps:
        if XlsccInfo in dep:
            transitive_runfiles.append(dep[XlsccInfo].cc_headers)
    runfiles = runfiles.merge_all(transitive_runfiles)
    return runfiles

def _get_transitive_built_files_for_xls_cc_ir(ctx):
    """Returns the transitive built files from a 'xls_cc_ir' ctx.

    Args:
      ctx: The current rule's context object.

    Returns:
      The transitive built files from a 'xls_cc_ir' ctx.
    """
    transitive_built_files = []

    transitive_built_files.append(ctx.attr.src[DefaultInfo].files)
    if ctx.attr.block:
        transitive_built_files.append(ctx.attr.block[DefaultInfo].files)
    transitive_built_files.append(ctx.attr._xlscc_tool[DefaultInfo].files)
    transitive_built_files.append(ctx.attr
        ._default_cc_header_files[DefaultInfo].files)
    transitive_built_files.append(ctx.attr
        ._default_synthesis_header_files[DefaultInfo].files)

    for dep in ctx.attr.src_deps:
        transitive_built_files.append(dep[DefaultInfo].files)

    if not transitive_built_files:
        return None

    return transitive_built_files

def _xls_cc_ir_impl(ctx):
    """The implementation of the 'xls_cc_ir' rule.

    Converts a C/C++ source file to an IR file.

    Args:
      ctx: The current rule's context object.

    Returns:
      A tuple with the following elements in the order presented:
        1. The IrFileInfo provider
        1. The ConvIrInfo provider
        1. The list of built files.
        1. The runfiles.
        1. The XlsccInfo provider (transitive C++ headers)
    """
    XLSCC_FLAGS = (
        "module_name",
        "block",
        "block_pb_out",
        "block_pb_text",
        "top",
        "package",
        "clang_args_file",
        "defines",
        "include_dirs",
        "meta_out",
        "meta_out_text",
        "block_from_class",
        "z3_rlimit",
        "generate_new_fsm",
        "merge_states",
        "split_states_on_channel_ops",
        "debug_ir_trace_loop_context",
        "debug_ir_trace_loop_control",
        "debug_print_fsm_states",
        "channel_strictness",
        "default_channel_strictness",
        "max_unroll_iters",
        "print_optimization_warnings",
        "debug_write_function_slice_graph_path",
    )

    xlscc_args = append_default_to_args(
        ctx.attr.xlscc_args,
        _DEFAULT_XLSCC_ARGS,
    )

    include_dirs = depset(transitive = [
        dep[XlsccIncludeInfo].include_dir
        for dep in ctx.attr.src_deps
        if XlsccIncludeInfo in dep
    ])

    # Append to user paths.
    xlscc_args["include_dirs"] = (
        xlscc_args.get("include_dirs", "") + "," + ",".join(include_dirs.to_list()) + ",${PWD},./," +
        ctx.genfiles_dir.path + "," + ctx.bin_dir.path + "," +
        "xls/contrib/xlscc/synth_only," +
        "xls/contrib/xlscc/synth_only/ac_compat," +
        ctx.attr._default_cc_header_files.label.workspace_root  # This must the last directory in the list.
    )

    # Append to user defines.
    xlscc_args["defines"] = (
        xlscc_args.get("defines", "") + "__SYNTHESIS__,__xlscc__," +
        "__AC_OVERRIDE_OVF_UPDATE_BODY=,__AC_OVERRIDE_OVF_UPDATE2_BODY="
    )

    is_args_valid(xlscc_args, XLSCC_FLAGS)
    my_args = args_to_string(xlscc_args)

    ir_filename = get_output_filename_value(
        ctx,
        "ir_file",
        ctx.attr.name + _IR_FILE_EXTENSION,
    )
    ir_file = ctx.actions.declare_file(ir_filename)
    outputs = [ir_file]
    block_pb_out_filename = getattr(ctx.attr, "block_pb_out")
    block_from_class_name = getattr(ctx.attr, "block_from_class")
    block_from_class_flag = ""
    if block_from_class_name:
        block_pb_file = ctx.actions.declare_file(block_pb_out_filename.name)
        outputs.append(block_pb_file)
        block_pb = block_pb_file.path
        block_from_class_flag = "--block_from_class {}".format(block_from_class_name)
    else:
        block_pb = ctx.file.block.path

    meta_out_flag = ""
    metadata_out_filename = getattr(ctx.attr, "metadata_out")
    if metadata_out_filename:
        meta_pb_file = ctx.actions.declare_file(metadata_out_filename.name)
        outputs.append(meta_pb_file)
        meta_out_flag = "--meta_out " + meta_pb_file.path

    meta_out_text_flag = ""
    metadata_out_text = getattr(ctx.attr, "meta_out_text")
    if metadata_out_text:
        meta_out_text_flag = "--meta_out_text"

    function_slice_graph_out_flag = ""
    debug_write_function_slice_graph_filename = getattr(ctx.attr, "debug_write_function_slice_graph_path")
    if debug_write_function_slice_graph_filename:
        function_slice_graph_file = ctx.actions.declare_file(debug_write_function_slice_graph_filename.name)
        outputs.append(function_slice_graph_file)
        function_slice_graph_out_flag = "--debug_write_function_slice_graph_path " + function_slice_graph_file.path

    # Get runfiles
    runfiles = _get_runfiles_for_xls_cc_ir(ctx)

    cc_headers = _get_headers_for_xls_cc_ir(ctx)

    xlscc_log_filename = get_output_filename_value(
        ctx,
        "xlscc_log_file",
        ctx.attr.name + ".xlscc.log",
    )
    log_file = ctx.actions.declare_file(xlscc_log_filename)
    outputs.append(log_file)

    ctx.actions.run_shell(
        outputs = outputs,
        # The IR converter executable is a tool needed by the action.
        tools = [ctx.executable._xlscc_tool],
        # The files required for converting the C/C++ source file.
        inputs = runfiles.files,
        command = "set -o pipefail; {} {} --block_pb {} {} {} {} {} {} 2>&1 >{} | tee {}".format(
            ctx.executable._xlscc_tool.path,
            ctx.file.src.path,
            block_pb,
            block_from_class_flag,
            meta_out_flag,
            meta_out_text_flag,
            function_slice_graph_out_flag,
            my_args,
            ir_file.path,
            log_file.path,
        ),
        mnemonic = "CompileXLSCC",
        progress_message = "Converting %s" % ir_file.short_path,
    )
    return [
        IrFileInfo(ir_file = ir_file),
        ConvIrInfo(original_input_files = _get_xls_cc_ir_source_files(ctx), ir_interface = None),
        outputs,
        runfiles,
        XlsccInfo(cc_headers = cc_headers),
    ]

_xls_cc_ir_attrs = {
    "src": attr.label(
        doc = "The C/C++ source file containing the top level block. A " +
              "single source file must be provided. The file must have a '" +
              _CC_FILE_EXTENSION + "' extension.",
        mandatory = True,
        allow_single_file = [_CC_FILE_EXTENSION],
    ),
    "block": attr.label(
        doc = "Protobuf describing top-level block interface. A single " +
              "source file must be provided. The file " + "must have a '" +
              _PROTOBIN_FILE_EXTENSION + "' or a '" +
              _PROTOTEXT_FILE_EXTENSION + "' or a '" +
              _TEXTPROTO_FILE_EXTENSION + "' or a '" +
              _BINARYPB_FILE_EXTENSION + "' extension. To create this " +
              "protobuf automatically from your C++ source file, use " +
              "'block_from_class' instead. Exactly one of 'block' or " +
              "'block_from_class' should be specified.",
        mandatory = False,
        allow_single_file = [
            _PROTOBIN_FILE_EXTENSION,
            _BINARYPB_FILE_EXTENSION,
            _PROTOTEXT_FILE_EXTENSION,
            _TEXTPROTO_FILE_EXTENSION,
        ],
    ),
    "block_pb_out": attr.output(
        doc = "Output protobuf describing top-level block interface.",
        mandatory = False,
    ),
    "block_from_class": attr.string(
        doc = "Filename of the generated top-level block interface protobuf, " +
              "created from a C++ class. To manually specify this protobuf, " +
              "use 'block' instead. Exactly one of 'block' or " +
              "'block_from_class' should be specified.",
        mandatory = False,
    ),
    "src_deps": attr.label_list(
        doc = "Additional source files for the rule. The file must have a " +
              _CC_FILE_EXTENSION + ", " + _H_FILE_EXTENSION + " or " +
              _INC_FILE_EXTENSION + " extension.",
        allow_files = [
            _CC_FILE_EXTENSION,
            _H_FILE_EXTENSION,
            _INC_FILE_EXTENSION,
        ],
    ),
    "xlscc_args": attr.string_dict(
        doc = "Arguments of the XLSCC conversion tool.",
    ),
    "ir_file": attr.output(
        doc = "Filename of the generated IR. If not specified, the " +
              "target name of the bazel rule followed by an " +
              _IR_FILE_EXTENSION + " extension is used.",
    ),
    "metadata_out": attr.output(
        doc = "Filename of the generated metadata protobuf",
        mandatory = False,
    ),
    "meta_out_text": attr.bool(
        doc = "Whether the generated metadata protobuf should output as a text protobuf",
        default = False,
        mandatory = False,
    ),
    "debug_write_function_slice_graph_path": attr.output(
        doc = "Filename of the generated function slice graph",
        mandatory = False,
    ),
    "_xlscc_tool": attr.label(
        doc = "The target of the XLSCC executable.",
        default = Label("//xls/contrib/xlscc:xlscc"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_default_cc_header_files": attr.label(
        doc = "Default C/C++ header files for xlscc.",
        default = Label("@com_github_hlslibs_ac_types//:ac_types_as_data"),
        cfg = "target",
    ),
    "_default_synthesis_header_files": attr.label(
        doc = "Default synthesis header files for xlscc.",
        default = Label("//xls/contrib/xlscc:synth_only_headers"),
        cfg = "target",
    ),
    "xlscc_log_file": attr.output(
        doc = "The filename to log stderr to. If not specified, 'xlscc.log' is used.",
    ),
}

def _xls_cc_ir_impl_wrapper(ctx):
    """The implementation of the 'xls_cc_ir' rule.

    Wrapper for xls_cc_ir_impl. See: xls_cc_ir_impl.

    Args:
      ctx: The current rule's context object.

    Returns:
      IrFileInfo provider
      ConvIrInfo provider
      DefaultInfo provider
      XlsccInfo provider
    """
    ir_result, ir_conv_info, built_files, runfiles, xlscc_info = _xls_cc_ir_impl(ctx)
    return [
        ir_result,
        ir_conv_info,
        DefaultInfo(
            files = depset(
                direct = built_files,
                transitive = _get_transitive_built_files_for_xls_cc_ir(ctx),
            ),
            runfiles = runfiles,
        ),
        xlscc_info,
    ]

xls_cc_ir = rule(
    doc = """A build rule that converts a C/C++ source file to an IR file.

Examples:

1) A simple IR conversion example. Assume target 'a_block_pb' is
defined.

```
    xls_cc_ir(
        name = "a_ir",
        src = "a.cc",
        block = ":a_block_pb",
    )
```
    """,
    implementation = _xls_cc_ir_impl_wrapper,
    attrs = dicts.add(
        _xls_cc_ir_attrs,
        CONFIG["xls_outs_attrs"],
        xls_toolchain_attrs,
    ),
)

def xls_cc_ir_macro(
        name,
        src,
        block = None,
        block_pb_out = None,
        block_from_class = None,
        src_deps = [],
        xlscc_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        metadata_out = None,
        **kwargs):
    """A macro that instantiates a build rule generating an IR file from a C/C++ source file.

    The macro instantiates a rule that converts a C/C++ source file to an IR
    file and the 'enable_generated_file_wrapper' function. The generated files
    are listed in the outs attribute of the rule.

    Examples:

    1) A simple IR conversion example. Assume target 'a_block_pb' is defined.

    ```
        xls_cc_ir(
            name = "a_ir",
            src = "a.cc",
            block = ":a_block_pb",
        )
    ```

    Args:
      name: The name of the rule.
      src: The C/C++ source file containing the top level block. A single source
        file must be provided. The file must have a '.cc' extension.
      block: Protobuf describing top-level block interface. A single source file
        single source file must be provided. The file must have a '.protobin'
        or a '.binarypb' extension. To create this protobuf automatically from
        your C++ source file, use 'block_from_class' instead. Exactly one of
        'block' or 'block_from_class' should be specified.
      block_pb_out: Protobuf describing top-level block interface, same as block,
        but an output used with block-from-class.
      block_from_class: Filename of the generated top-level block interface
        protobuf created from a C++ class. To manually specify this protobuf,
        use 'block' instead. Exactly one of 'block' or 'block_from_class'
        should be specified.
      src_deps: Additional source files for the rule. The file must have a
        '.cc', '.h' or '.inc' extension.
      xlscc_args: Arguments of the XLSCC conversion tool.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      metadata_out: Generated metadata proto.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    string_type_check("src", src)
    string_type_check("block", block, True)
    string_type_check("block_pb_out", block, True)
    string_type_check("block_from_class", block_from_class, True)
    list_type_check("src_deps", src_deps)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    if block == None and block_from_class == None:
        fail("Either 'block' or 'block_from_class' is required.")

    if block != None and block_from_class != None:
        fail("Specify either 'block' or 'block_from_class', but not both.")

    # Append output files to arguments.
    kwargs = _append_xls_cc_ir_generated_files(kwargs, name)

    xls_cc_ir(
        name = name,
        src = src,
        block = block,
        block_from_class = block_from_class,
        block_pb_out = block_pb_out,
        src_deps = src_deps,
        xlscc_args = xlscc_args,
        outs = _get_xls_cc_ir_generated_files(kwargs),
        metadata_out = metadata_out,
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def _xls_cc_verilog_impl(ctx):
    """The implementation of the 'xls_cc_verilog' rule.

    Converts a C/C++ file to an IR, optimizes the IR, and generates a verilog
    file from the optimized IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      ConvIrInfo provider.
      OptIrArgInfo provider.
      CodegenInfo provider.
      DefaultInfo provider.
    """
    cc_ir_result, ir_conv_info, ir_conv_built_files, ir_conv_runfiles, _xlscc_info = _xls_cc_ir_impl(ctx)
    opt_ir_result, ir_opt_info, opt_ir_built_files, opt_ir_runfiles = xls_ir_opt_ir_impl(
        ctx,
        cc_ir_result,
        ir_conv_info.original_input_files,
    )
    codegen_info, verilog_built_files, verilog_runfiles = xls_ir_verilog_impl(
        ctx,
        opt_ir_result,
        ir_conv_info,
    )
    runfiles = ir_conv_runfiles.merge_all([opt_ir_runfiles, verilog_runfiles])
    return [
        ir_conv_info,
        ir_opt_info,
        codegen_info,
        DefaultInfo(
            files = depset(
                direct = ir_conv_built_files + opt_ir_built_files +
                         verilog_built_files,
                transitive = _get_transitive_built_files_for_xls_cc_ir(ctx),
            ),
            runfiles = runfiles,
        ),
    ]

_cc_verilog_attrs = dicts.add(
    _xls_cc_ir_attrs,
    xls_ir_opt_ir_attrs,
    xls_ir_verilog_attrs,
    CONFIG["xls_outs_attrs"],
    xls_toolchain_attrs,
)

xls_cc_verilog = rule(
    doc = """A build rule that generates a Verilog file from a C/C++ source file.

Examples:

1) A simple example. Assume target 'a_block_pb' is defined.

```
    xls_cc_verilog(
        name = "a_verilog",
        src = "a.cc",
        block = ":a_block_pb",
        codegen_args = {
            "generator": "combinational",
            "module_name": "A",
            "top": "A_proc",
        },
    )
```
    """,
    implementation = _xls_cc_verilog_impl,
    attrs = _cc_verilog_attrs,
)

def xls_cc_verilog_macro(
        name,
        src,
        block,
        verilog_file,
        src_deps = [],
        xlscc_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from a C/C++ source file.

    The macro instantiates a build rule that generates a Verilog file from
    a DSLX source file. The build rule executes the core functionality of
    following macros:

    1. xls_cc_ir (converts a C/C++ file to an IR),
    1. xls_ir_opt_ir (optimizes the IR), and,
    1. xls_ir_verilog (generated a Verilog file).

    Examples:

    1) A simple example. Assume target 'a_block_pb' is defined.

    ```
        xls_cc_verilog(
            name = "a_verilog",
            src = "a.cc",
            block = ":a_block_pb",
            codegen_args = {
                "generator": "combinational",
                "module_name": "A",
                "top": "A_proc",
            },
        )
    ```

    Args:
      name: The name of the rule.
      src: The C/C++ source file containing the top level block. A single source
        file must be provided. The file must have a '.cc' extension.
      block: Protobuf describing top-level block interface. A single source file
        single source file must be provided. The file must have a '.protobin'
        , '.pbtxt', or a '.binarypb' extension.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' extension.
      src_deps: Additional source files for the rule. The file must have a
        '.cc', '.h' or '.inc' extension.
      xlscc_args: Arguments of the XLSCC conversion tool.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      codegen_args: Arguments of the codegen tool. For details on the arguments,
        refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    string_type_check("src", src)
    string_type_check("block", block)
    string_type_check("verilog_file", verilog_file)
    list_type_check("src_deps", src_deps)
    dictionary_type_check("xlscc_args", xlscc_args)
    dictionary_type_check("opt_ir_args", opt_ir_args)
    dictionary_type_check("codegen_args", codegen_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = _append_xls_cc_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_verilog_generated_files(kwargs, name, codegen_args)

    xls_cc_verilog(
        name = name,
        src = src,
        block = block,
        verilog_file = verilog_file,
        src_deps = src_deps,
        xlscc_args = xlscc_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        outs = _get_xls_cc_ir_generated_files(kwargs) +
               get_xls_ir_opt_ir_generated_files(kwargs) +
               get_xls_ir_verilog_generated_files(kwargs, codegen_args) +
               [verilog_file],
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )
