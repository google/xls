# Copyright 2023 The XLS Authors
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
This module contains functions for testing the build rules for XLS.

"""

def _scheduling_args_proto(ctx):
    """The implementation of the 'scheduling_args_proto' rule.

    Outputs a text protobuf with scheduling options to provide to other rules.

    Args:
      ctx: The current rule's context object.
    Returns:
      A file with the text protobuf representation of the scheduling arguments provided.
    """
    ctx.actions.run(
        outputs = [ctx.outputs.scheduling_options_proto],
        executable = ctx.executable._generate_protos,
        arguments = [
            "--delay_model={}".format(ctx.attr.delay_model),
            "--pipeline_stages={}".format(ctx.attr.pipeline_stages),
            "--worst_case_throughput=6",
            "--mutual_exclusion_z3_rlimit=-1",
            "--use_fdo=false",
            "--fdo_iteration_number=2",
            "--fdo_refinement_stochastic_ratio=1.0",
            "--fdo_path_evaluate_strategy=window",
            "--scheduling_options_used_textproto_file={}".format(
                ctx.outputs.scheduling_options_proto.path,
            ),
        ],
    )
    return [
        DefaultInfo(files = depset(direct = [ctx.outputs.scheduling_options_proto])),
    ]

scheduling_args_proto = rule(
    doc = """A build rule that generates a text protobuf with scheduling options.

Example:

    ```
    scheduling_args_proto(
        name = "some_proto_scheduling",
        delay_model = "unit",
        pipeline_stages = 2,
        scheduling_options_proto = "some_proto_scheduling.txtproto",
    )
    ```
    """,
    implementation = _scheduling_args_proto,
    attrs = {
        "scheduling_options_proto": attr.output(),
        "delay_model": attr.string(),
        "pipeline_stages": attr.int(),
        "_generate_protos": attr.label(
            default = Label("//xls/tools:generate_options_protos_main"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def _codegen_args_proto(ctx):
    """The implementation of the 'codegen_args_proto' rule.

    Outputs a text protobuf with codegen options to provide to other rules.

    Args:
      ctx: The current rule's context object.
    Returns:
      A file with the text protobuf representation of the codegen arguments provided.
    """
    if ctx.attr.is_pipelined:
        generator = "GENERATOR_KIND_PIPELINE"
    else:
        generator = "GENERATOR_KIND_COMBINATIONAL"
    ctx.actions.run(
        outputs = [ctx.outputs.codegen_options_proto],
        executable = ctx.executable._generate_protos,
        arguments = [
            "--generator={}".format("pipeline" if ctx.attr.is_pipelined else "combinational"),
            "--top={}".format(ctx.attr.top),
            "--module_name={}".format(ctx.attr.module_name),
            "--reset_data_path={}".format(str(ctx.attr.reset_data_path).lower()),
            "--register_merge_strategy={}".format(ctx.attr.register_merge_strategy),
            "--input_valid_signal=input_valid",
            "--output_valid_signal=output_valid",
            "--flop_inputs",
            "--flop_inputs_kind=flop",
            "--flop_outputs",
            "--flop_outputs_kind=skid",
            "--flop_single_value_channels",
            "--add_idle_output",
            "--reset=rst_n",
            "--reset_active_low",
            "--reset_asynchronous=false",
            "--use_system_verilog=false",
            "--gate_recvs",
            "--array_index_bounds_checking",
            "--streaming_channel_data_suffix=_data",
            "--streaming_channel_valid_suffix=_vld",
            "--streaming_channel_ready_suffix=_rdy",
            "--codegen_options_used_textproto_file={}".format(
                ctx.outputs.codegen_options_proto.path,
            ),
        ],
    )
    return [
        DefaultInfo(files = depset(direct = [ctx.outputs.codegen_options_proto])),
    ]

codegen_args_proto = rule(
    doc = """A build rule that generates a text protobuf with codegen options.

Example:

    ```
    codegen_args_proto(
        name = "some_proto_codegen",
        codegen_options_proto = "some_proto_codegen.txtproto",
        is_pipelined = True,
        module_name = "main",
        reset_data_path = False,
        top = "__some__main",
    )
    ```
    """,
    implementation = _codegen_args_proto,
    attrs = {
        "codegen_options_proto": attr.output(),
        "top": attr.string(),
        "module_name": attr.string(),
        "reset_data_path": attr.bool(),
        "is_pipelined": attr.bool(),
        "register_merge_strategy": attr.string(default = "IdentityOnly"),
        "_generate_protos": attr.label(
            default = Label("//xls/tools:generate_options_protos_main"),
            executable = True,
            cfg = "exec",
        ),
    },
)
