# XLS Pipeline scheduling

[TOC]

Pipeline scheduling divides the IR nodes of an XLS function or proc into a
sequence of stages constituting a feed-forward pipeline. Sequential stages are
separated by registers enabling pipeline parallelism. The schedule must satisfy
dependency constraints between XLS nodes as well as timing constraints imposed
by the target clock frequency. Pipeline scheduling has multiple competing
optimization objectives: minimize number of stages (minimize pipeline latency),
minimize maximum delay of any stage (maximize clock frequency), and minimize the
number of pipeline registers.

## Scheduling process

Pipeline scheduling occurs in two phases:

1.  Determine the effective clock period. This clock period defines the maximum
    delay, based on XLS's internal [delay model](delay_estimation.md), through
    any pipeline stage and limits how many IR operations might be placed in each
    stage.

1.  Given the constraints of the effective clock period and, optionally, a
    user-defined number of pipeline stages, find the schedule which minimizes
    the number of pipeline registers. Pipeline registers are required for any IR
    operation whose value which is used in a later stage.

The schedule process is controlled via several options defined
[here](https://github.com/google/xls/tree/main/xls/scheduling/pipeline_schedule.h). These
options are typically passed in as flags to the
[`codegen_main` binary](https://github.com/google/xls/tree/main/xls/tools/codegen_main.cc)
but maybe set programmatically. Each is optional though at least one of **clock
period** or **pipeline stages** must be specified. Different combinations of
options result in different strategies as described [below](#common-options).

Clock period : The target clock period.

Pipeline stages : The number of stages in the pipeline.

Clock margin percent : The percentage to reduce the target clock period before
scheduling. May only be specified with **clock period**. This option is
equivalent to specifying a reduced value for **clock period**.

Clock period relaxation percent : This is the percentage that the computed
minimum clock period, as determined by the number of pipeline stages, is
increased (relaxed) prior to scheduling. May not be specified with **clock
period**.

### Step 1: determine the effective clock period

The effective clock period determines the maximum delay through any pipeline
stage for the purpose of scheduling. The value is determined in one of two ways
depending upon whether the **clock period** option is specified.

1.  **clock period** specified

    The effective clock period is set the **clock period** value. If **clock
    margin percent** is also specified, then the effective clock period is also
    reduced by the given percentage. Example: if **clock period** is 800ps and
    **clock margin percent** is 20% then the effective clock period is 640ps.

1.  **clock period** not specified

    In this case, **pipeline stages** must be specified. The effective clock
    period is computed as the minimum clock period in which a schedule may be
    found that meets timing with the specified number of pipeline stages. This
    is done via a binary search through clock period values, where at each step
    of the binary search the scheduler is run in its entirety. If **clock period
    relaxation percent** is specified then the computed effective clock period
    is *increased* by the given percentage. The motivation is that this
    relaxation may result in fewer pipeline registers because of increased
    scheduling flexibility. Example: if the minimum clock period found by XLS
    was 1000ps and **clock period relaxation percent** is 10% the effective
    clock period is 1100ps.

### Step 2: schedule to minimize pipeline registers

Once an effective clock period is determined, XLS computes a schedule which
minimizes the number of registers (see [below](#sdc) for details) while
satisfying various constraints, including the critical path delay constraints
imposed by the effective clock period. The number of stages in the pipeline may
be specified by the user via the **pipeline stages** option. If the number of
pipeline stages specified is too small an error such that no feasible schedule
can be found then an error is returned. If **pipeline stages** is not given then
the minimum number of stages which meets the delay constraint imposed by the
effective clock period is used.

### Options for common scheduling objectives {#common-options}

Different scheduling options result in different optimization strategies for the
scheduler. Below are several common scheduling objectives and options which
should be set to enable them.

1.  Minimize the number of pipeline registers for a given clock period and given
    number of pipeline stages.

    Specify both **clock period** and **pipeline stages**. The scheduler will
    attempt to minimize the number of pipeline registers given those
    constraints. The option **clock margin percent** can be swept to search the
    local design space (or equivalently, sweep **clock period**)

1.  Minimize the clock period for a given number of pipeline stages

    Specify only **pipeline stages**. XLS will find a schedule with minimum
    clock period with a secondary objective of minimizing the number of pipeline
    registers. Sweeping **clock period relaxation percent** explores relaxing
    the timing constraint which may result in fewer pipeline registers.

1.  Minimize the number of pipeline stages for a given clock period

    Specify only **clock period**. XLS will find a schedule of the minimum
    number of stages with a secondary objective of minimizing the number of
    pipeline registers. The option **clock margin percent** can be swept to
    search the local design space (or equivalently, sweep **clock period**)

1.  Minimize the number of pipeline registers for a given clock period

    Specify only **clock period** and sweep **pipeline stages**. Pick the
    schedule which produces the minimum number of pipeline registers.

1.  Sweep the entire scheduling space

    The various options directly or indirectly control the two degrees of
    freedom within the scheduler: pipeline stages and clock period. Sweeping
    these two degrees of freedom is most easily done by sweeping **pipeline
    stages** and **clock period relaxation percent**. The advantage of sweeping
    **clock period relaxation percent** instead of **clock period** directly is
    that the percent relaxation can be a fixed range (e.g., 0 to 50%) for all
    designs and each value will produce a feasible schedule. If **clock period**
    is swept some combinations of **pipeline stages** and **clock period**
    values will result in an error returned because the design point is
    infeasible.

## Minimizing pipeline registers via SDC scheduling {#sdc}

For scheduling pipelines, XLS uses a variation on the approach described in
[SDC-Based Modulo Scheduling for Pipeline Synthesis](https://www.csl.cornell.edu/~zhiruz/pdfs/sdcmod-iccad2013.pdf).
The basic principle is to create a set of real-valued variables, each
corresponding to the cycle in which a node is scheduled or the
lifetime[^lifetime] of a node, and then carefully constrain the variables using
linear inequality constraints such that minimizing a linear objective always
gives an answer with *integer* values for all the variables. This avoids the
need for integer linear programming, which is NP complete, and instead can be
solved with linear programming, which is polynomial time in theory and takes
roughly cubic time in practice.

Prior to the implementation of the SDC scheduler, we used a scheduler based on
taking the min-cut of the node graph with Ford-Fulkerson. However, this design
proved difficult to extend with needed features like IO constraints, and unlike
the SDC algorithm was not optimal in the particular, narrow, sense that it
assigns nodes to cycles such that the required register bits are minimized. We
found that switching from the min-cut algorithm to SDC resulted in marginal
improvements to benchmarks and increased compile times by an small and
acceptable amount.

### Constraints

Currently, we generate a variety of constraints:

-   Causality constraints, i.e.: if node Y uses the output of node X, then the
    cycle of node X must be less than or equal to the cycle of node Y.
-   Timing constraints, i.e.: if the critical path between node X and node Y is
    greater than the clock period, then the cycle of Y must be strictly greater
    than the cycle of X.
-   IO constraints among sends and receives on a given channel (see the codegen
    documentation for more details).
-   "Node in cycle" constraints, which allow forcing a given node to be
    scheduled in a given cycle. This is useful for incremental scheduling in the
    scheduling pass pipeline.
-   "Receives first, sends last" constraints, which allow accessing the old
    behavior in which receives all went into the first cycle and sends all went
    into the last cycle.
-   Backedge constraints: when the initiation interval is 1, a state parameter
    and its corresponding next state must be scheduled in the same cycle. More
    generally, we build up a graph of states where there is an edge between two
    states if the output of one affects the input of another, and then compute
    the strongly connected components of this graph. All nodes within a strongly
    connected component must be in the same cycle.

### Additional technical details

The linear inequality constraints can be summarized by a matrix M and a vector y
such that Mx ≤ y. If the linear program has the property described above (that
minimizing a linear objective gives an integer answer), then the matrix M is
considered to be *integral*. One class of integral matrices is that of the
"totally unimodular matrices". The exact definition of this class is out of
scope to discuss here, but it suffices to say that it includes constraints of
the following form:

-   Difference constraints between variables with an integer bound: `x - y ≤ k`
    where `k` is an integer
-   Constraints of the form `x - y - z ≤ k` where `k` is an integer

In the SDC scheduler, we use `x - y ≤ k` constraints to express causality and
timing constraints, whereas `x - y - z ≤ k` constraints are used to constrain
the lifetime variables to be equal to the difference between the max user cycle
and the cycle of a given node.

[^lifetime]: The lifetime of a node is the interval starting at the cycle number
    assigned to the node and ending at the maximum cycle number of the
    users of the node.
