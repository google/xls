# XLS Tutorials

The XLS team has written several tutorials to help explain language features and
design techniques. Here they are, grouped by topic:

## DSLX

*   [Hello, XLS!](hello_xls.md) : A brief introduction to writing and evaluating
    your first design in DSLX.
*   [Float-to-int conversion](float_to_int.md) : A guide to writing "real" logic
    in DSLX, demonstrated by creating an IEEE-754 binary32, i.e., C `float` to
    `int32_t` converter.
*   [Intro to parametrics](intro_to_parametrics.md) : A demonstration on how
    functions and types can be parameterized to allow a single implementation to
    apply to many different data layouts.
*   [`for` expressions](crc32.md) : Explains how to understand and write looping
    constructs in DSLX.
*   [`enumerate` and `match` expressions](prefix_scan.md) : Explains how to use
    `enumerate()` expressions to control loop iteration and how to use the
    `match` pattern-matching expression for selecting between alternatives.
*   [What is a proc?](what_is_a_proc.md): A step-by-step introduction to procs,
    XLS's answer to how to write modules with state and explicit communication
    interfaces.
*   [How to use procs](how_to_use_procs.md) (communicating sequential
    processes): Provides a basic introduction to writing stateful and
    communicating modules, i.e., procs.
*   [Dataflow & Time in XLS](dataflow_and_time.md) (theoretical musings):
    Discusses the relationship between dataflow and execution order in XLS, and
    how DSLX is a "timeless" language.
