# Releases

The following tools are packaged in the
[nightly releases](https://github.com/google/xls/releases):

<!-- LINT.IfChange -->
-   [`parse_and_typecheck_dslx_main`](tools.md#parse-and-typecheck-dslx-main):
    Parses and typechecks DSLX (`.x`) source files and emits errors.
-   [`dslx_fmt`](tools.md#dslx-fmt): Auto-formats DSLX source files.
-   [`interpreter_main`](tools.md#interpreter-main): Interprets DSLX source
    files and runs tests.
-   [`prove_quickcheck_main`](tools.md#prove-quickcheck-main): Proves DSLX
    `#[quickcheck]` properties over all inputs using an SMT solver.
-   [`ir_converter_main`](tools.md#ir-converter-main): Converts DSLX source
    files into [XLS IR](ir_semantics.md) (`.ir`).
-   [`opt_main`](tools.md#opt-main): Runs XLS IR through the
    [optimization](optimizations/optimizations.md) pipeline.
-   [`check_ir_equivalence_main`](tools.md#check-ir-equivalence-main): Verifies
    that two XLS IR files are logically equivalent.
-   [`delay_info_main`](tools.md#delay-info-main): Reports per-node
    [delay estimates](delay_estimation.md) and critical-path timing for XLS IR.
-   [`codegen_main`](tools.md#codegen-main): Schedules and lowers XLS IR into
    (System)Verilog.
-   [`print_bom`](tools.md#print-bom): Summarizes bill-of-materials (BOM)
    metrics from `codegen_main --block_metrics_path` outputs.
-   [`proto_to_dslx_main`](tools.md#proto-to-dslx-main): Converts a Protocol
    Buffer message instance and its accompanying schema into a DSLX module.
-   [`dslx_ls`](tools.md#dslx-ls): Provides Language Server Protocol (LSP)
    support for DSLX.
<!-- LINT.ThenChange(../dist/BUILD) -->

See the [tools listing](tools.md) page for additional tools buildable from
source.
