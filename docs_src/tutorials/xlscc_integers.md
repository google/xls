# [xls/contrib] Tutorial: XLS[cc] arbitrary width integers.

[TOC]

This tutorial is aimed at walking you through usage of arbitrary width integers
using the XlsInt class.

Note that an `ac_datatypes` compatibility layer is also provided so that the
same C++ code can be used for both simulation and XLS[cc] synthesis with just a
change in the include paths:
[ac_compat](https://github.com/google/xls/tree/main/xls/contrib/xlscc/synth_only/ac_compat).

## C++ Source

## Introduction to fixed-width integers.

XLS also provides a template class for fixed-width integers. These are declared
using the template class `XlsInt<int Width, bool Signed = true>`.

To utilize fixed with integer types, include
[xls_int.h](https://github.com/google/xls/tree/main/xls/contrib/xlscc/synth_only/xls_int.h).

Create a `test.cc` with the following contents for the rest of this tutorial.

```c++
#include "xls_int.h"

#pragma hls_top
XlsInt<55, true> foo(XlsInt<17, false> x, XlsInt<5, false> y) {
    return x+y;
}
```

## Configuring XLS[cc] for fixed-width integers.

XLS[cc] fixed-width integers have a dependency on the `ac_datatypes` library.
Clone the repository (https://github.com/hlslibs) into a directory named
`ac_datatypes`.

```shell
git clone https://github.com/hlslibs/ac_types.git ac_datatypes
```

Then create the a `clang.args` file with the following contents to configure the
include paths and pre-define the `__SYNTHESIS__` name as a macro.

```
-D__SYNTHESIS__
-I/path/to/your/xls/contrib/xlscc/synth_only
-I/path/containing/ac_datatypes/..
```

## Translate into optimized XLS IR.

Now that the C++ function has been created, `xlscc` can be used to translate the
C++ into XLS IR. `opt_main` is used afterwards to optimize and transform the IR
into a form more easily synthesized into verilog.

```
$ ./bazel-bin/xls/contrib/xlscc/xlscc test.cc --clang_args_file clang.args > test.ir
$ ./bazel-bin/xls/tools/opt_main test.ir > test.opt.ir
```

The resulting `test.opt.ir` file should look something like the following

```
package my_package

file_number 1 "/usr/local/google/home/seanhaskell/tmp/tutorial_int.cc"
file_number 2 "/usr/local/google/home/seanhaskell/xls/xls/contrib/xlscc/synth_only/xls_int.h"

top fn foo(x: bits[17], y: bits[5]) -> bits[55] {
  literal.132: bits[1] = literal(value=0, id=132, pos=[(1,6,2)])
  literal.133: bits[13] = literal(value=0, id=133, pos=[(1,6,2)])
  xid4: bits[18] = concat(literal.132, x, id=134, pos=[(1,6,2)])
  yid6__1: bits[18] = concat(literal.133, y, id=135, pos=[(1,6,2)])
  literal.141: bits[37] = literal(value=0, id=141, pos=[(1,6,2)])
  xid4id8: bits[18] = add(xid4, yid6__1, id=140, pos=[(1,6,2)])
  ret xid4id8id2: bits[55] = concat(literal.141, xid4id8, id=143, pos=[(1,6,2)])
}
```

## Additional XLS[cc] examples.

For developers, it is possible to check if a specific feature is supported by
checking
[xls_int_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/xls_int_test.cc)
for unit tests.
