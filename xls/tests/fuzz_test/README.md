# DSLX Fuzz Testing

This directory contains examples/integration tests for DSLX fuzz testing.

See https://google.github.io/xls/dslx_reference/#fuzz-tests for more
information.

## Supported Domains

The following domain specifications are supported in the `domains` argument of
the `#[fuzz_test]` attribute:

-   **Arbitrary**: `()` - Explores the full range of the type.
-   **Numeric Range**: `Type:min..max` - Explores values in the range `[min,
    max)`. Example: `u32:0..100`. "End-inclusive" ranges work too, e.g.,
    `Type:min..=max` explores values in the range `[min, max]`.
-   **Element Of**: `[val1, val2, ...]` - Explores only the listed values.
    Example: `[u32:5, 10, 15]`.
-   **Tuples**: `(Domain1, Domain2, ...)` - For tuple parameters. Example:
    `(u32:0..10, [u8:1, 2])`. The parentheses are required.

## Known Limitations

-   **Array parameters**: are not yet supported.
-   **Struct parameters**: (e.g., `StructName { field: Domain, ... }`) are not
    yet supported.
-   Fuzzing is currently limited to types that can be mapped to native C++ types
    up to 64 bits for full specialization. Larger types fallback to `xls::Value`
    and may have limited mutation capabilities.
