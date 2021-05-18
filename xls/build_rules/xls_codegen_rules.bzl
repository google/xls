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

load("//xls/build_rules:xls_ir_rules.bzl", "ir_common_attrs")

CodegenInfo = provider(
    doc = "A provider containing Codegen file information for the target. It " +
          "is created and returned by the codegen rule.",
    fields = {
        "verilog_file": "File: The Verilog file.",
        "module_sig_file": "File: The module signature of the Verilog file.",
        "schedule_file": "File: The schedule of the module.",
    },
)

ir_to_codegen_attrs = {
    "codegen_args": attr.string_dict(
        doc = "Arguments of the codegen tool.",
    ),
    "verilog_file": attr.output(
        doc = "The Verilog file generated.",
    ),
    "_codegen_tool": attr.label(
        doc = "The target of the codegen executable.",
        default = Label("//xls/tools:codegen_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

def ir_to_codegen_impl(ctx, src):
    """The core implementation of the 'ir_to_codegen' rule.

    Generates a Verilog file.

    Args:
      ctx: The current rule's context object.
      src: The source file.
    Returns:
      CodegenInfo provider
      DefaultInfo provider
    """
    my_generated_files = []

    # default arguments
    codegen_args = ctx.attr.codegen_args
    codegen_flags = ctx.actions.args()
    codegen_flags.add(src.path)
    codegen_flags.add("--delay_model", codegen_args.get("delay_model", "unit"))

    # parse arguments
    CODEGEN_FLAGS = (
        "clock_period_ps",
        "pipeline_stages",
        "delay_model",
        "entry",
        "generator",
        "input_valid_signal",
        "output_valid_signal",
        "manual_load_enable_signal",
        "flop_inputs",
        "flop_outputs",
        "module_name",
        "clock_margin_percent",
        "reset",
        "reset_active_low",
        "reset_asynchronous",
        "use_system_verilog",
    )
    verilog_file = None
    module_sig_file = None
    schedule_file = None
    for flag_name in codegen_args:
        if flag_name in CODEGEN_FLAGS:
            codegen_flags.add("--{}".format(flag_name), codegen_args[flag_name])
            if flag_name == "generator" and codegen_args[flag_name] == "combinational":
                # Pipeline generator produces a schedule artifact.
                schedule_file = ctx.actions.declare_file(
                    ctx.attr.name + ".schedule.textproto",
                )
                my_generated_files.append(schedule_file)
                codegen_flags.add("--output_schedule_path", schedule_file.path)
        else:
            fail("Unrecognized argument: %s." % flag_name)

    verilog_file = ctx.actions.declare_file(ctx.attr.name + ".v")
    module_sig_file = ctx.actions.declare_file(ctx.attr.name + ".sig.textproto")
    my_generated_files += [verilog_file, module_sig_file]
    codegen_flags.add("--output_verilog_path", verilog_file.path)
    codegen_flags.add("--output_signature_path", module_sig_file.path)

    ctx.actions.run(
        outputs = my_generated_files,
        tools = [ctx.executable._codegen_tool],
        inputs = [src, ctx.executable._codegen_tool],
        arguments = [codegen_flags],
        executable = ctx.executable._codegen_tool.path,
        mnemonic = "Codegen",
        progress_message = "Building Verilog file: %s" % (verilog_file.path),
    )
    return [
        CodegenInfo(
            verilog_file = verilog_file,
            module_sig_file = module_sig_file,
            schedule_file = schedule_file,
        ),
        DefaultInfo(
            files = depset(my_generated_files),
        ),
    ]

def _ir_to_codegen_impl_wrapper(ctx):
    """The implementation of the 'ir_to_codegen' rule.

    Wrapper for ir_to_codegen_impl. See: ir_to_codegen_impl.

    Args:
      ctx: The current rule's context object.
    Returns:
      See: codegen_impl.
    """
    return ir_to_codegen_impl(ctx, ctx.file.src)

ir_to_codegen = rule(
    doc = """A build rule that generates a Verilog file.

        Examples:

        1) A file as the source.

        ```
            ir_to_codegen(
                name = "a_ir_to_codegen",
                src = "a.ir",
            )
        ```

        2) A target as the source.

        ```
            ir_opt(
                name = "a",
                src = "a.ir",
            )
            ir_to_codegen(
                name = "a_ir_to_codegen",
                src = ":a",
            )
        ```
    """,
    implementation = _ir_to_codegen_impl_wrapper,
    attrs = dict(ir_common_attrs.items() + ir_to_codegen_attrs.items()),
)
