# DSLX: "Domain Specific Language: X"

Think: a Domain Specific Language (DSL) called "X".

## Directory structure map

*   [`bytecode`](https://https://github.com/google/xls/tree/main/xls/dslx/fmt/): DSLX
    bytecode interpreter, used in interpreting constexpr AST expressions as well
    as quick-feedback execution (without needing to convert to XLS IR)
*   [`cpp_transpiler`](https://https://github.com/google/xls/tree/main/xls/dslx/cpp_transpiler/):
    Library facilities that convert DSLX types to their C++ equivalents
*   [`fmt`](https://https://github.com/google/xls/tree/main/xls/dslx/fmt/): DSLX
    auto-formatting facilities and tests
*   [`ir_convert`](https://https://github.com/google/xls/tree/main/xls/dslx/ir_convert/):
    Routines for converting typechecked DSLX programs into XLS IR
*   [`lsp`](https://https://github.com/google/xls/tree/main/xls/dslx/lsp/): DSLX server
    that implements the Language Server Protocol
*   [`run_routines`](https://https://github.com/google/xls/tree/main/xls/dslx/run_routines/):
    Library routines used in running DSLX programs and tests, e.g. from the main
    runner binary
*   [`stdlib`](https://https://github.com/google/xls/tree/main/xls/dslx/stdlib/): DSLX
    standard library modules -- these are intended to be included in any
    distribution as they include many very common facilities
*   [`tests`](https://https://github.com/google/xls/tree/main/xls/dslx/tests/): DSLX
    language-level tests, e.g. often for individual features or cross products
    of features in more "unit level" fashion than whole programs
*   [`type_system`](https://https://github.com/google/xls/tree/main/xls/dslx/type_system/):
    DSLX type system / type inference library
