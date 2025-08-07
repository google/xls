# XLS: IR Pass Profiling

The XLS Optimization pipeline is quite complex with many passes run repeatedly
and in fixed-point loops. This can make pinpointing what parts of optimization
are taking time difficult. To improve this situation we have a
`--passes_profile=<path>` flag on most binaries which interact with the pass
system. This flag will generate a (uncompressed) [pprof
profile](https://github.com/google/pprof/blob/main/doc/README.md) recording the
number of invocations, number of changed runs, and timings of each pass.

## Using pprof

There are many pprof visualizers including the `go` visualizer found at
[google/pprof](https://github.com/google/pprof).

## Known issues

Due to how we structure our pass pipelines some compound passes might appear
multiple times in a stack. This is due to the profile recording both the actual
compound pass and a wrapper which is used to implement some opt-level behaviors.
