# Tutorial: Hello, XLS!

So, you're interested in learning more about XLS and DSLX! Super! This tutorial
is aimed at the very basics of getting started with XLS: getting your execution
environment set up and running the most trivial of DSLX examples: printing the
standard "Hello, world!" message to the terminal.

Yes, even though XLS is a hardware synthesis language, it still needs basic
printing and string support, if only for debugging!

## 1. Installation and building

First things first: if you haven't yet done so, download the XLS sources from
Github:

```
git clone https://github.com/google/xls.git xls
```

Next, build the project tree. XLS includes several dependencies that can take a
while to build, so the first build may take a while; subsequent builds will be
much shorter.

> **NOTE**: If you don't have Bazel installed, install it: check the Bazel
> website for instructions. The other prerequisites are a C++20-compliant
> compiler toolchain and a Python3 interpreter; check with your distribution for
> installation instructions for both.

Start the XLS build by running:

```
bazel build -c opt xls/...
```

Then go get a cup of coffee. LLVM and Z3 are big projects, and will take a while
to compile (but only the first time). Binary releases are coming soon: they'll
avoid the need for long local compiles.

## 2. Create your module

With your toolchain built, let's get to coding! Open up an editor and create a
file called `hello_xls.x` in your XLS checkout root directory. Populate it with
the following:

```dslx
fn hello_xls(hello_string: u8[11]) {
  trace!(hello_string);
}
```

Let's go over this, line-by-line:

1.  This first line declares a fn (`fn`) named "`hello_xls`". This function
    accepts an array of eleven characters (u8) called `hello_string`, and
    returns no value (the return type would be specified after the argument
    list's closing parenthesis and before the function-opening curly brace, if
    the function returned a value).
2.  This second line invokes the built-in `trace!` directive, passing it the
    function's input string, and throws away the result.

## 3. Say hello, XLS!

Let's run (and test) our code!

First thing, though, we should make sure our module parses and passes type
checking. The fastest way to do that is via the DSLX
"[repl](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)",
conveniently called
[`repl`](https://github.com/google/xls/tree/main/xls/dev_tools/repl.cc). You can run it
against the above example with the command:

```
$ ./bazel-bin/xls/tools/repl hello_xls.x
```

This tool first examines the specified module for language correctness, and will
print an `INVALID_ARGUMENT` error if it fails to parse or typecheck. In that
case, fix the errors and type `:reload` to try again. `repl` supports other
features (IR, Verilog, and LLVM code examination), but those are outside the
scope of this tutorial.

Once you have a parsing DSLX file, the best way to "smoke test" a module is via
the
[DSLX interpreter](https://github.com/google/xls/tree/main/xls/dslx/interpreter_main.cc).
First, though, we need a test case for it to execute. Add the following to the
end of your `hello_xls.x` file:

```dslx-snippet
#[test]
fn hello_test() {
  hello_xls("Hello, XLS!")
}
```

Again, going line-by-line:

1.  This directive tells the interpreter that the next function is a test
    function, meaning that it shouldn't be passed down the synthesis chain and
    that it should be executed by the interpreter.
2.  This line declares the [test] function `hello_test`, which takes no args and
    returns no value.
3.  The only line in this function invokes the `hello_xls` function and passes
    it a chipper greeting.

With both the function and its corresponding test/driver in place, let's fire it
up! Open a terminal and execute the following in the XLS checkout root
directory:

```
$ ./bazel-bin/xls/dslx/interpreter_main hello_xls.x
```

You should see the following output:

```
[ RUN UNITTEST  ] hello_test
trace of hello_string @ hello.x:4:17-4:31: [72, 101, 108, 108, 111, 44, 32, 88, 76, 83, 33]
[            OK ]
[===============] 1 test(s) ran; 0 failed; 0 skipped.
```

Perfect! While this may not be what you initially expected, examine the output
elements carefully: they correspond to the ASCII codes of the characters in
"Hello, XLS!" When designing and debugging hardware, signals are more often
numbers than strings, which is why they're represented as numbers here.

Congrats! You've written your first piece of hardware in DSLX! It might be more
satisfying, though, if your hardware _actually did anything_. For that, see the
next tutorial,
[float-to-int conversion](../tutorials/float_to_int.md).
