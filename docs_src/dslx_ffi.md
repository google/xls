# DSLX FFI interfacing with Verilog Modules

Sometimes it might be useful to instantiate existing Verilog modules from DSLX.
This could be for various reasons; sometimes there is an existing code-base with
specific optimizations one wants to use.

The concept of calling an external implementation from some language is
typically referred to as Foreign Function Interface or short FFI, used below
for brevity.

DSLX can interface with combinational Verilog modules; [sequential FFI] is
planned.

## Foreign Function Interface in DSLX

Every external module to interface will need to have a DSLX implementation
as a function with an annotation that tells DSLX the module instantiation
at code-generation time. The DSLX implementation will be used in the interpreter
and JIT, for instance in tests.

### Simple Example

Suppose you'd like to instantiate the following Verilog module in a DSLX design:

```verilog
module myfoo #(
  parameter int N = 32
)(
  input wire [N-1:0] x,
  output wire [N-1:0] out
);
  assign out = x + 1;
endmodule
```

First, write a DSLX function, that has the relevant inputs and and return value,
as well as a implementation that is functionally equivalent. Here, there is one
`input` parameter, mapped to a function parameter, and one `output` parameter,
mapped to the return of the DSLX function:

```dslx
fn foo(a: u32) -> u32 { a + u32:1 }
```

You can now add an annotation `#[extern_verilog("...")]` to the DSLX function
that contains a textual template for the instantiation that should happen. There
are placeholders in `{...}`-braces that will be replaced with the actual values
at code-generation time:

```dslx
#[extern_verilog("
myfoo {fn} (       // Placeholder for the instantiation name.
   .x({a}),        // Reference to name in function parameter.
   .out({return})  // Placeholder for the output.
);                 // Semicolon optional, code-generation will always add one.
")]
fn foo(a: u32) -> u32 {
    a + u32:1
}
```

Let's look at this in detail

*   Inside the `extern_verilog("...")`, you add the code that will be the
    Verilog instantiation of the particular module: this is just a regular
    module instantiation of the module `myfoo` that you'd like to interface
    with.
*   The `{fn}` placeholder is needed and will be expanded to the actual
    instantiation name decided at code-generation time.
*   The `{a}` placeholder references a value in the function prototype, in this
    case the parameter `a`.
*   The output parameter of the `myfoo` function is wired to the special value
    `{return}` which represents the return value of the function.
*   The types the module will receive are based on the type mentioned in the
    function prototype. Parameter `a` and the return value are both u32, so
    `.x()` and `.out()` will be connected to `wire [31:0]`'s at code generation
    time.

## Parameterization

DSLX functions allow [parameterization][parametric function] as do Verilog
modules. The same technique as above can be applied to parameters, referencing
values in the function prototype in the instantiation template in curly braces.
DSLX parameter values are const-evaluated and then provided in the textual
template under the given name. With this, you can now make full use of the
parametric properties of the Verilog module:

```dslx
#[extern_verilog("
myfoo {fn} #(
   .N({WIDTH})     // Expanded to the constant determined at compile-time.
)(
   .x({a}),
   .out({return})
)
")]
fn foo<WIDTH: u32>(a: bits[WIDTH]) -> bits[WIDTH] {
    a + uN[WIDTH]:1
}
```

This will now automatically parameterize the module instantiation with the same
parameter value the DSLX function is called.

## Mapping Aggregate types

The first example looked at a simple integer type for parameter and return
values, but it is also possible to refer to [tuples], another common way to
represent more complex data in DSLX. You can refer to values inside tuples in
the same way you'd do inside DSLX, with an index suffix:

```dslx
#[extern_verilog("
mybar {fn} (
   .x({a.0}),
   .y({a.1}),
   .z({b}),
   .someout({return.0}),
   .otherout({return.1})
)
")]
fn bar(a: (s32, s32), b: s32) -> (s32, s32) {
    (a.0 + a.1, b)
}
```

If you just access the tuple by its name (e.g. `a` in this case), the Verilog
module receives the bit-concatenated content of that tuple, `a.0 ++ a.1`.

In the `xls/examples` directory, you find a more complete [ffi example]
including nested tuples (`{return.0.1}`).

## Code Generation

The code generator needs to know the critical path delay of the Verilog module
to be able to do proper scheduling and pipelining. This information can be
provided by a [codegen parameter] `--ffi_fallback_delay_ps` (see BUILD file in
the [ffi example]).

There are plans for an automatic [ffi delay estimate].

## Tips and Tricks

The following examples are technically possible right now, but it should not
necessarily be considered a supported use-case.

Given that the Verilog template just accepts Verilog pasted into the output, you
can use Verilog features to do some transformations directly inside the template
while accessing the parameters from the DSLX function prototype:

```dslx
#[extern_verilog("
mybaz {fn} #(
   .WIDTH($bits({a}))              // Calling system functions
)(
   .modify((42)'({a})),            // Type casting
   .all_the_bits({{ {b}, {c} }}),  // Concatenate; note escaped braces.
   .out({return})
)
")]
fn baz(a: u32, b: u32, c: u32) -> u32 {
    u32:42  // local implementation
}
```

Note that the Verilog concatenation needs to use curly braces, but since these
are 'special' characters within the textual template, they need to be escaped.
This is done by doubling them up: `{{...}}` will result in `{...}` in the
code-generated output.

In this particular example for the system function it would probably be a good
idea to const-evaluate expressions as part of the [parametric function]
parameters, then pass this constant.

Even the following will work: create a local wire and assignments that we
assemble from parameters to the template; here, we use that to adapt the `wire
[42:0]` output of `myquux` to whatever our return type is:

```dslx
#[extern_verilog("

wire [42:0] {return}_adapted_to_module;

myquux {fn} (
   .x(a)
   .out({return}_adapted_to_module)
);

assign {return} = ({RESULT_BITS})'({return}_adapted_to_module);
")]
fn quux<RESULT_BITS: u32>(a: u32) -> uN[RESULT_BITS] {
    a as uN[RESULT_BITS]
}
```

Of course, at that point, XLS can't guarantee anymore that wire identifiers are
unique. Handle this rope with care :)

[tuples]: ./dslx_reference.md#tuple-type
[delay model]: ./delay_estimation.md
[codegen parameter]: ./codegen_options.md#pipelining-and-scheduling-options
[parametric function]: ./dslx_reference.md#parametric-functions
[Sequential FFI]: https://github.com/google/xls/issues/1301
[ffi delay estimate]: https://github.com/google/xls/issues/1399
[ffi example]: https://github.com/google/xls/tree/main/xls/examples/ffi.x
