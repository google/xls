# XLS FAQ

## Q: How do I select bubble strategies in pipeline generation?

**Tags:** codegen, pipeline, configuration

I/O behavior is described in the
[codegen options](codegen_options.md#io-behavior) documentation. ("codegen" for
XLS is the concept of "exactly what verilog should be produced?") There are
currently options for controlling queueing behavior at the inputs or outputs of
the block, but not internal to the pipeline, ability to stall with queueing that
is embedded in the generated pipeline is tracked in
[issue \#255](https://github.com/google/xls/issues/255).

## Q: What is the granularity of ready/valid signaling?

**Tags:** codegen, pipeline, configuration

Ready/valid signals are associated with a (streaming) channel, and in general
I/O signaling is configured on a per-channel basis. Users are expected to send
things "broadside" (all together at once) if they should share the same
ready/valid signaling; e.g. by sending a struct or array over the channel.

See the `--streaming_channel_*` options within the
[codegen options](codegen_options.md#naming) documentation.

## Q: How do I call my XLS functions from C++?

**Tags:** native, simulation, cpp

The steps are:

1.  Wrap up your XLS so it can be called from C++ (using a utility).
2.  Include the created header.
3.  Call the "Run" API with XLS-understood values.

The
[`cc_xls_ir_jit_wrapper` rule](https://google.github.io/xls/bazel_rules_macros/#cc_xls_ir_jit_wrapper)
in the Bazel rule set invokes a tool (the
[JIT wrapper generator](https://github.com/google/xls/tree/main/xls/jit/jit_wrapper_generator_main.py))
that makes a shim that helpfully JIT compiles the IR to native code (e.g. x64
code), and provides an object that can be used as a C++ callable.

As an example, see the
[float32 multiply test](https://github.com/google/xls/tree/main/xls/dslx/stdlib/float32_mul_test.cc),
which calls `Run()` on the float32 multiplier which is written in XLS and
wrapped in the
[`float32_mul_jit_wrapper` build target](https://github.com/google/xls/tree/main/xls/dslx/stdlib/BUILD).

Note that for some APIs, e.g. those taking a XLS single precision float, the
created interface *will* be able to accept a native C++ `float` directly, and
similar for types like `uint32_t`, `uint64_t`, etc.
