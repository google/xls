# xls/contrib/eco

Engineering Change Order (ECO) tooling for XLS IR. Given two versions of an IR package — a golden *before* and a revised *after* — this computes the minimum set of node/edge edits that turns one into the other and replays them on the golden IR.

The motivation is **design reuse.** When a small spec change lands after the IR has already been scheduled, codegened, or verified, ECO lets you keep node identities, stage assignments, and downstream artifacts for the unchanged majority of the design — only the diff has to be re-examined. The patch is a serialized proto, so the diff itself is reviewable and replayable.

## Quick start (recommended: Bazel)

The full diff → apply → verify chain is wrapped as two macros in [eco_build_defs.bzl](eco_build_defs.bzl). `xls_patch_ir` invokes `//xls/dev_tools:check_ir_equivalence_main` automatically, so the target only builds if the patched IR is provably equivalent to the after IR.

```
xls_dslx_ir_diff(
    name = "mydiff",
    srcs = [":before.x", ":after.x"],
    dslx_top = "main",
)

xls_patch_ir(
    name = "mypatched",
    ir_diff = ":mydiff",
)
```

`xls_dslx_ir_diff` compiles both DSLX inputs with `xls_dslx_opt_ir`, runs `ged_main`, and emits `.patch.bin` and `.report.txt` plus an `EcoPatchInfo` provider. `xls_patch_ir` consumes that, runs `patch_ir_main`, then runs the equivalence check and writes an `.equiv.report`. Optional kwargs: `schedule=`, `activation_count=`, `top=`, `timeout=` (for `xls_patch_ir`) and the MCS / GED tuning flags from the tables below (for `xls_dslx_ir_diff`). [test/BUILD](test/BUILD) uses both as the end-to-end harness.

## Manual

1. Diff the two IRs to produce a patch.

   ```
   ged_main --before_ir=a.ir --after_ir=b.ir --patch=patch.bin
   ```

2. Apply the patch to the before IR.

   ```
   patch_ir_main --input_ir_path=a.ir --input_patch_path=patch.bin --output_ir_path=patched.ir
   ```

3. Verify the patched IR is equivalent to the after IR.

   ```
   check_ir_equivalence_main patched.ir b.ir
   ```

Step 3 is the trust boundary — until `check_ir_equivalence_main` ([//xls/dev_tools:check_ir_equivalence_main](../../dev_tools)) passes, the patch is unverified. To preserve the original pipeline schedule across step 2, pass `--input_schedule_path=schedule.textproto` (a `PackageScheduleProto`); deleted nodes drop out, substituted nodes keep their stage. For procs, step 3 needs `--activation_count=N`. Tuning flags for step 1 are in the MCS / GED tables below.

## Pipeline

1. **IR → graph.** [xls_ir_to_graph.cc](xls_ir_to_graph.cc) lowers `xls::Package` to an `XLSGraph` ([graph.h](graph.h)) carrying op / type / literal / state-element attributes and structural signatures (`signature = hash(label, ordered incoming labels, unordered outgoing labels)`).
2. **MCS preprocessing** ([mcs.cc](mcs.cc); see below) finds the maximum common subgraph. Boundary pairs — MCS nodes adjacent to the residual — are *pinned*; interior pairs are *cut*.
3. **GED** ([ged.cc](ged.cc); see below) computes minimum edit cost on the residual plus pinned nodes. [ged_cost_functions.cc](ged_cost_functions.cc) forces pinned nodes to map to their MCS partner.
4. **Patch.** [ir_patch_gen.cc](ir_patch_gen.cc) serializes the `GEDResult` as an `IrPatchProto` ([ir_patch.proto](ir_patch.proto)); [patch_ir.cc](patch_ir.cc) replays it against a live `xls::Package`.

## MCS

Redundancy-Reduced Backtracking search (RRSplit) [[1]](#ref-1), adapted to directed/labeled IR graphs: initial candidate classes are partitioned by exact label + signature + in/out degree, and adjacency tests are exact forward/backward edge-index profiles. The pass is what gives the chain its scalability — large unchanged regions never enter the GED search at all.

`ged_main` flags that drive MCS:

| Flag | Default | Effect |
| --- | --- | --- |
| `--use_mcs` | `true` | Run MCS preprocessing before GED. Set `false` to skip it. |
| `--mcs_cutoff` | `-1` | Stop MCS as soon as residual unmatched nodes ≤ N. `-1` runs to completion. |
| `--mcs_optimal` | `true` | If `false`, MCS may terminate on a no-improvement plateau (faster, may leave a smaller MCS). |
| `--mcs_timeout` | `-1` | MCS wall-clock cap in seconds. `-1` disables. |

## GED

Exact best-first GED search [[2]](#ref-2). Each search state's lower bound is a rectangular Linear Sum Assignment over the `(m+n)×(m+n)` substitution / deletion / insertion / dummy cost matrix, solved by a shortest-augmenting-path algorithm [[3]](#ref-3) in [lap_solver.cc](lap_solver.cc).

`ged_main` flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `--before_ir`, `--after_ir` | — | IR file paths. Positional fallback if both are unset. |
| `--timeout` | `-1` | GED wall-clock cap in seconds. `0` returns the initial LSAP-derived solution. `-1` disables. |
| `--optimal` | `false` | Require optimal GED. Overrides `--timeout`. |
| `--patch` | `""` | Output `IrPatchProto` path. |
| `--report` | `""` | Output stats report: graph sizes, MCS runtime + match counts + prune %, GED runtime, op breakdown, peak RSS. |

## Logging

Both binaries route through `xls::InitXls`, which honours absl's logging flags:

- `--v=0` (default): top-level progress — start, MCS enabled, GED finished, totals.
- `--v=1`: parsed graph sizes, residual sizes after MCS preprocessing, pinned boundary counts.
- `--v=2`: per-pair MCS pinning, per-edit substitute/insert/delete with node names.
- `--vmodule=mcs=2,ged=1`: scope verbosity per source file.
- `--logtostderr`: emit to stderr (default for CLI use).

The `--report` file from `ged_main` is the structured counterpart to log output and is meant for CI / regression dashboards.

## Tests

- [mcs_test.cc](mcs_test.cc) — RRSplit internals.
- [xls_ir_to_graph_test.cc](xls_ir_to_graph_test.cc) — IR → graph lowering.
- [test/BUILD](test/BUILD) — end-to-end DSLX pairs (`crc32`, `apfloat_fmac`, `riscv_simple`, `histogram`, `vector_core`, `fir_filter_*`) wired through `xls_patch_ir`, so the build fails if the patched IR isn't equivalent.

## Limitations

Channels aren't graph nodes, so revisions that change channel types (e.g. `apfloat_fmac.rev1`) need a manual fixup before `patch_ir_main` produces equivalent IR. Recipe is in [test/BUILD](test/BUILD) above the commented `apfloat_patched_opt_ir` target.

## References

<a id="ref-1"></a>[1] Kaiqiang Yu, Kaixin Wang, Cheng Long, Laks V.S. Lakshmanan, and Reynold Cheng. 2025. Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking Approach. *Proc. ACM Manag. Data* 3, 3, Article 160 (2025). https://doi.org/10.1145/3725404

<a id="ref-2"></a>[2] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, and Patrick Martineau. 2015. An Exact Graph Edit Distance Algorithm for Solving Pattern Recognition Problems. In *Proc. 4th Int. Conf. on Pattern Recognition Applications and Methods (ICPRAM)*. https://hal.archives-ouvertes.fr/hal-01168816

<a id="ref-3"></a>[3] David F. Crouse. 2016. On Implementing 2D Rectangular Assignment Algorithms. *IEEE Transactions on Aerospace and Electronic Systems* 52, 4 (2016), 1679–1696.
