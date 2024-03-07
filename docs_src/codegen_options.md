# Codegen Options

This document outlines some useful knobs for running codegen on an XLS design.
Codegen is the process of generating RTL from IR and is where operations are
scheduled and mapped into RTL constructs. The output of codegen is suitable for
simulation or implementation via standard tools that understand Verilog or
SystemVerilog.

[TOC]

# Input specification

-   `<input.ir>` is a positional argument giving the path to the ir file.
-   `--top` specifies the top function or proc to codegen.
-   `--codegen_options_proto=...` specifies the filename of a protobuf
    containing the arguments to supply codegen other than the scheduling
    arguments. Details can be found in codegen_flags.cc
-   `--scheduling_options_proto=...` specifies the filename of a protobuf
    containing the scheduling arguments. Details can be found in
    scheduling_options_flags.cc

# Output locations

The following flags control where output files are put. In addition to Verilog,
codegen can generate files useful for understanding or integrating the RTL.

-   `--output_verilog_path` is the path to the output Verilog file.
-   `--output_schedule_path` is the path to a textproto that shows into which
    pipeline stage the scheduler put IR ops.
-   `--output_schedule_ir_path` is the path to the "IR" representation of the
    design, a post-scheduling IR that includes any optimizations or transforms
    during the scheduling pipeline.
-   `--output_block_ir_path` is the path to the "block IR" representation of the
    design, a post-scheduling IR that is timed and includes registers, ports,
    etc.
-   `--output_signature_path` is the path to the signature textproto. The
    signature describes the ports, channels, external memories, etc.
-   `--output_verilog_line_map_path` is the path to the verilog line map
    associating lines of verilog to lines of IR.
-   `--codegen_options_used_textproto_file` is the path to write a textproto
    containing the actual configuration used for codegen.

# Pipelining and Scheduling Options

The following flags control how XLS maps IR operations to RTL, and if applicable
control the scheduler.

-   `--generator=...` controls which generator to use. The options are
    `pipeline` and `combinational`. The `pipeline` generator runs a scheduler
    that partitions the IR ops into pipeline stages.
-   `--delay_model=...` selects the delay model to use when scheduling. See the
    [page here](delay_estimation.md) for more detail.
-   `--clock_period_ps=...` sets the target clock period. See
    [scheduling](scheduling.md) for more details on how scheduling works. Note
    that this option is optional, without specifying clock period XLS will
    estimate what the clock period should be.
-   `--pipeline_stages=...` sets the number of pipeline stages to use when
    `--generator=pipeline`.
-   `--clock_margin_percent=...` sets the percentage to reduce the target clock
    period before scheduling. See [scheduling](scheduling.md) for more details.
-   `--period_relaxation_percent=...` sets the percentage that the computed
    minimum clock period is increased. May not be specified with
    `--clock_period_ps`.
-   `--minimize_clock_on_error` is enabled by default. If enabled, when
    `--clock_period_ps` is given with an infeasible clock (in the sense that XLS
    cannot pipeline this input for this clock, even with other constraints
    relaxed), XLS will find and report the minimum feasible clock period if one
    exists. If disabled, XLS will report only that the clock period was
    infeasible, potentially saving time.
-   `--worst_case_throughput=...` sets the worst-case throughput bound to use
    when `--generator=pipeline`. If set, allows scheduling a pipeline with
    worst-case throughput no slower than once per N cycles (assuming no stalling
    `recv`s). If not set, defaults to 1.

    NOTE: If set to 0 or a negative value, no throughput minimum will be
    enforced.

-   `--additional_input_delay_ps=...` adds additional input delay to the inputs.
    This can be helpful to meet timing when integrating XLS designs with other
    RTL.

-   `--ffi_fallback_delay_ps=...` Delay of foreign function calls if not
    otherwise specified. If there is no measurement or configuration for the
    delay of an invoked modules, this is the value used in the scheduler.

-   `--io_constraints=...` adds constraints to the scheduler. The flag takes a
    comma-separated list of constraints of the form `foo:send:bar:recv:3:5`
    which means that sends on channel `foo` must occur between 3 and 5 cycles
    (inclusive) before receives on channel `bar`. Note that for a constraint
    like `foo:send:foo:send:3:5`, no constraint will be applied between a node
    and itself; i.e.: this means all *different* pairs of nodes sending on `foo`
    must be in cycles that differ by between 3 and 5. If the special
    minimum/maximum value `none` is used, then the minimum latency will be the
    lowest representable `int64_t`, and likewise for maximum latency. For an
    example of the use of this, see
    [this example](https://github.com/google/xls/tree/main/xls/examples/constraint.x) and
    the associated BUILD rule.

-   `explain_infeasibility` configures what to do if scheduling fails. If set,
    the scheduling problem is reformulated with extra slack variables in an
    attempt to explain why scheduling failed.

-   `infeasible_per_state_backedge_slack_pool` If specified, the specified value
    must be > 0. Setting this configures how the scheduling problem is
    reformulated in the case that scheduling fails. If specified, this value
    will cause the reformulated problem to include per-state backedge slack
    variables, which increases the complexity. This value scales the objective
    such that adding slack to the per-state backedge is preferred up until total
    slack reaches the pool size, after which adding slack to the shared backedge
    slack variable is preferred. Increasing this value should give more specific
    information about how much slack each failing backedge needs at the cost of
    less actionable and harder to understand output.

-   `--scheduling_options_used_textproto_file` is the path to write a textproto
    containing the actual configuration used for scheduling.

# Feedback-driven Optimization (FDO) Options

The following flags control the feedback-driven optimizations in XLS. For now,
an iterative SDC scheduling method is implemented, which can take low-level
feedbacks (typically from downstream tools, e.g., OpenROAD) to guide the delay
estimation refinements in XLS. For now, FDO is disabled by default
(`--use_fdo=false`).

-   `--use_fdo=true/false` Enable FDO. If false, then the `--fdo_*` options are
    ignored.
-   `--fdo_iteration_number=...` The number of FDO iterations during the
    pipeline scheduling. Must be an integer >= 2.
-   `--fdo_delay_driven_path_number=...` The number of delay-driven subgraphs in
    each FDO iteration. Must be a non-negative integer.
-   `--fdo_fanout_driven_path_number=...` The number of fanout-driven subgraphs
    in each FDO iteration. Must be a non-negative integer.
-   `--fdo_refinement_stochastic_ratio=...` \*path_number over
    refinement_stochastic_ratio paths are extracted and \*path_number paths are
    randomly selected from them for synthesis in each FDO iteration. Must be a
    positive float <= 1.0.
-   `--fdo_path_evaluate_strategy=...` Path evaluation strategy for FDO.
    Supports path, cone, and window.
-   `--fdo_synthesizer_name=...` Name of synthesis backend for FDO. Only
    supports yosys.
-   `--fdo_yosys_path=...` Absolute path of yosys.
-   `--fdo_sta_path=...` Absolute path of OpenSTA.
-   `--fdo_synthesis_libraries=...` Synthesis and STA libraries.

# Naming

Some names can be set at codegen via the following flags:

-   `--module_name=...` sets the name of the generated verilog module.
-   For functions, `--input_valid_signal=...` and `--output_valid_signal=...`
    adds and sets the name of valid signals when `--generator` is set to
    `pipeline`.
-   `--manual_load_enable_signal=...` adds and sets the name of an input that
    sets the load-enable signals of each pipeline stage.
-   For procs, `--streaming_channel_data_suffix=...`,
    `--streaming_channel_valid_suffix=...`, and
    `--streaming_channel_ready_suffix=...` set suffixes to be used on their
    respective signals in ready/valid channels. For example,
    `--streaming_channel_valid_suffix=_vld` for a channel named `ABC` would
    result in a valid port called `ABC_vld`.

# Reset Signal Configuration

-   `--reset=...` sets the name of the reset signal. If not specified, no reset
    signal is used.
-   `--reset_active_low` sets if the reset is active low or high. Active high by
    default.
-   `--reset_asynchronous` sets if the reset is synchronous or asynchronous
    (synchronous by default).
-   `--reset_data_path` sets if the datapath should also be reset. True by
    default.

# Codegen Mapping

-   `--use_system_verilog` sets if the output should use SystemVerilog
    constructs such as SystemVerilog array assignments, `@always_comb`,
    `@always_ff`, asserts, covers, etc. True by default.
-   `--separate_lines` causes every subexpression to be emitted on a separate
    line. False by default.
-   `--multi_proc` causes every proc to be codegen'd.
-   `max_trace_verbosity` is the maximum verbosity allowed for traces. Traces
    with higher verbosity are stripped from codegen output. 0 by default.

## Format Strings

For some XLS ops, flags can override their default codegen behavior via format
string. These format strings use placeholders to fill in relevant information.

-   `--gate_format=...` sets the format string for `gate!` ops. Supported
    placeholders are:

    -   `{condition}`: Identifier (or expression) of the gate.
    -   `{input}`: Identifier (or expression) for the data input of the gate.
    -   `{output}`: Identifier for the output of the gate.
    -   `{width}`: The bit width of the gate operation.

    For example, consider a format string which instantiates a particular custom
    AND gate for gating:

    ```
    my_and gated_{output} [{width}-1:0] (.Z({output}), .A({condition}), .B({input}))
    ```

    And the IR gate operation is:

    `the_result: bits[32] = gate(the_cond, the_data)`

    This results in the following emitted Verilog:

    `my_and gated_the_result [32-1:0] (.Z(the_result), .A(the cond),
    .B(the_data));`

    To ensure valid Verilog, the instantiated template must declare a value
    named `{output}` (e.g. `the_result` in the example).

-   `--assert_format=...` sets the format string for assert statements.
    Supported placeholders are:

    -   `{message}`: Message of the assert operation.
    -   `{condition}`: Condition of the assert.
    -   `{label}`: Label of the assert operation. It is an error not to use the
        `label` placeholder.
    -   `{clk}`: Name of the clock signal. It is an error not to use the `clk`
        placeholder.
    -   `{rst}`: Name of the reset signal. It is an error not to use the `rst`
        placeholder.

    For example, the format string:

    ``{label}: `MY_ASSERT({condition}, "{message}")``

    could result in the following emitted Verilog:

    ``my_label: `MY_ASSERT(foo < 8'h42, "Oh noes!");``

-   `--smulp_format=...` and `--umulp_format=...` set the format strings for
    `smulp` and `umulp` ops respectively. These ops perform partial (or split)
    multiplies. Supported placeholders are:

    -   `{input0}` and `{input1}`: The two inputs.
    -   `{input0_width}` and `{input1_width}`: The width of the two inputs
    -   `{output}`: Name of the output. Partial multiply IP generally produces
        two outputs with the property that the sum of the two outputs is the
        product of the inputs. `{output}` should be the concatenation of these
        two outputs.
    -   `{output_width}`: Width of the output.

    For example, the format string:

    ```
    multp #(
        .x_width({input0_width}),
        .y_width({input1_width}),
        .z_width({output_width}>>1)
      ) {output}_inst (
        .x({input0}),
        .y({input1}),
        .z0({output}[({output_width}>>1)-1:0]),
        .z1({output}[({output_width}>>1)*2-1:({output_width}>>1)})])
      );
    ```

    could result in the following emitted Verilog:

    ```
    multp #(
      .x_width(16),
      .y_width(16),
      .z_width(32>>1)
    ) multp_out_inst (
      .x(lhs),
      .y(rhs),
      .z0(multp_out[(32>>1)-1:0]),
      .z1(multp_out[(32>>1)*2-1:(32>>1)])
    );
    ```

    Note the arithmetic performed on `output_width` to make the two-output
    `multp` block fill the concatenated output expected by XLS.

# I/O Behavior

-   `--flop_inputs` and `--flop_outputs` control if inputs and outputs should be
    flopped respectively. These flags are only used by the pipeline generator.

    For procs, inputs and outputs are channels with ready/valid signalling and
    have additional options controlling how inputs and outputs are registered.
    `--flop_inputs_kind=...` and `--flop_outputs_kind=...` flags control what
    the logic around the outputs and inputs look like respectively. The list
    below enumerates the possible kinds of output flopping and shows what logic
    is generated in each case.

    -   `flop`: Adds a pipeline stage at the beginning or end of the block to
        hold inputs or outputs. This is essentially a single-element FIFO.

![Flop Outputs](./flop_outputs.svg)

```
-   `skid`: Adds a skid buffer at the inputs or outputs of the block. The
    skid buffer can hold 2 entries.
```

![Skid Buffer](./skid_buffer.svg)

```
-   `zerolatency`: Adds a zero-latency buffer at the beginning or end of the
    block. This is essentially a single-element FIFO with bypass.
```

![Zero Latency Buffer](./zero_latency_buffer.svg)

-   `--flop_single_value_channels` controls if single-value channels should be
    flopped.

-   `--add_idle_output` adds an additional output port named `idle`. `idle` is
    the NOR of:

    1.  Pipeline registers storing the valid bit for each pipeline stage.

    2.  All valid registers stored for the input/output buffers.

    3.  All valid signals for the input channels.

-   For functions, when `--generator` is set to `pipeline`, optional 'valid'
    logic can be added by using `--input_valid_signal=...` and
    `--output_valid_signal=...`, which also set the names for the valid I/O
    signals. This logic has no 'ready' signal and thus provides no backpressure.
    See also [Naming](#naming).

-   See also [Reset Signal Configuration](#reset-signal-configuration).

# RAMs (experimental)

XLS has experimental support for using proc channels to drive an external RAM.
For an example usage, see
[this delay](https://github.com/google/xls/tree/main/xls/examples/delay.x) implemented
with a single-port RAM
([modeled here](https://github.com/google/xls/tree/main/xls/examples/ram.x)). Note that
receives on the response channel must be conditioned on performing a read,
otherwise there will be deadlock.

The codegen option `--ram_configurations` takes a comma-separated list of
configurations in the format `ram_name:ram_kind[:kind-specific-configuration]`.
For a `1RW` RAM, the format is
`ram_name:1RW:req_channel_name:resp_channel_name[:latency]`, where latency is 1
if unspecified. For a `1RW` RAM, there are several requirements these channels
must satisfy:

-   The request channel must be a tuple type with 4 entries corresponding to
    `(addr, wr_data, we, re)`. All entries must have type `bits`, and `we` and
    `re` must be a single bit.
-   The response channel must be a tuple type with a single entry corresponding
    to `(rd_data)`. `rd_data` must have the same width as `wr_data`.

Instead of the normal channel ports, the codegen option will produce the
following ports:

-   `{ram_name}_addr`
-   `{ram_name}_wr_data`
-   `{ram_name}_we`
-   `{ram_name}_re`
-   `{ram_name}_rd_data`

Note that there are no ready/valid signals as RAMs have fixed latency. There is
an internal buffer to catch the response and apply backpressure on requests if
needed.

When using `--ram_configurations`, you should generally add a scheduling
constraint via `--io_constraints` to ensure the request-send and
response-receive are scheduled to match the RAM's latency.

# Optimization

-   `--gate_recvs` emits logic to gate the data value of a receive operation in
    Verilog. In the XLS IR, the receive operation has the semantics that the
    data value is zero when the predicate is `false`. Moreover, for a
    non-blocking receive, the data value is zero when the data is invalid. When
    set to true, the data is gated and has the previously described semantics.
    However, the latter does utilize more resource/area. Setting this value to
    false may reduce the resource/area utilization, but may also result in
    mismatches between IR-level evaluation and Verilog simulation.

-   `--array_index_bounds_checking`: With this option set, an out of bounds
    array access returns the maximal index element in the array. If this option
    is not set, the result relies on the semantics of out-of-bounds array access
    in Verilog which is not well-defined. Setting this option to `true` may
    result in more resource/area. Setting this value to `false` may reduce the
    resource/area utilization, but may also result in mismatches between
    IR-level evaluation and Verilog simulation.

-   `--mutual_exclusion_z3_rlimit` controls how hard the mutual exclusion pass
    will work to attempt to prove that sends and receives are mutually
    exclusive. Concretely, this roughly limits the number of `malloc` calls done
    by the Z3 solver, so the output should be deterministic across machines for
    a given rlimit.

-   `--default_next_value_z3_rlimit` controls how hard our scheduling passes
    will work to prove that state params are fully covered by their `next_value`
    nodes, so that we can skip special handling for the case where no
    `next_value` node triggers. This is purely an optimization; everything will
    work correctly even if this is disabled (omitted, or set to -1). Concretely,
    this roughly limits the number of `malloc` calls done by the Z3 solver, so
    the output should be deterministic across machines for a given rlimit.
