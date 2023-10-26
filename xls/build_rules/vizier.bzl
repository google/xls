load("//xls/build_rules:xls_build_defs.bzl", "xls_ir_verilog")
load("@rules_hdl//place_and_route:build_defs.bzl", "place_and_route")
load("@rules_hdl//synthesis:build_defs.bzl", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

def generate_compression_block_parameter_optimization(name, optimized_ir, cocotb_test, test_parser = "//xls/tools/dashboard:generic_parser", test_log_file = "/tmp/test_log.txt", opt_eng_iterations = 5, opt_eng_suggestions = 1, opt_eng_algorithm = "NSGA2"):
    """Generate rules responsible for optimizing parameters of (de)compression
    block written in DSLX. This is done through Vizier blackbox optimization
    interface.

    Arguments:

    * name - string used for codegen 'module_name' attribute, used for
             constructing names for the rules
    * optimized_ir - DSLX (de)compression block passed in form of a label
                     string pointing to the optimized IR of the design.
    * cocotb_test - path to cocotb test used for evaluating the performance of
                    (de)compression block
    * test_parser - label for passing the parser for test output
    * test_log_file - path to temporary log file with test output
    * opt_eng_iterations - max iteration count for the optimization engine
    * opt_eng_suggestions - max suggestion count per iteration
    * opt_eng_algorithm - algorithm for the optimization engine

    This macro instantiates the following rules:
    # │ rule            │ name                                            │ visibility
    ──┼─────────────────┼─────────────────────────────────────────────────┼────────────
    1.│ xls_ir_verilog  │ <name>_compression_block_verilog                │ private
    2.│ verilog_library │ <name>_compression_block_verilog_lib            │ private
    3.│ synthesize_rtl  │ <name>_compression_block_synth                  │ private
    4.│ place_and_route │ <name>_compression_block_place_and_route        │ default
    5.│ py_test         │ <name>_compression_block_cocotb_test            │ default
    5.│ vizier_optimize │ <name>_compression_block_parameter_optimization │ default

    Most of the generated rules require setting the following environment variables
    when building or running the generated rules:

    * PIPELINE_STAGES - Number of pipeline stages to use in verilog generation.
                        Decimal in string form.
    * CLOCK_PERIOD - The amount of time a single clock period lasts in nanoseconds.
                     Decimal in string form.
    * PLACEMENT_DENSITY - How densely cells should be packaged on the die during
                          global placement. Float in string form, between 0.0
                          and 1.0.
    * TARGET_DIE_UTILIZATION_PERCENTAGE - Sets the die area based on an estimated
                                          die area target utilization. Decimal
                                          in string form, between 0 and 100.

    Environment variables should be set for building/running every rule generated
    by this macro except `vizier_optimize`. This should be done in bazel command
    line with argument '--action_env VAR=value', e.g.:

        ```
        bazel build --action_env PIPELINE_STAGES=1
                    --action_env CLOCK_PERIOD=1
                    --action_env PLACEMENT_DENSITY=0.8
                    --action_env TARGET_DIE_UTILIZATION_PERCENTAGE=30
                      //xls/modules/rle:rle_enc_compression_block_cocotb_test
        ```

    Rule `vizier_optimize` is a top-level rule used for executing vizier
    optimization framework. The rule starts `vizier_runner` python script
    responsible for running the optimization engine on given set of parameters,
    performance metrics and DSLX design. It doesn't need specifying environment
    viariables.
    """

    xls_ir_verilog(
        name = name + "_compression_block_verilog",
        src = optimized_ir,
        codegen_args = {
            "module_name": name,
            "delay_model": "unit",
            "pipeline_stages": "$PIPELINE_STAGES",
            "reset": "rst",
            "use_system_verilog": "false",
            "streaming_channel_data_suffix": "_data",
        },
        verilog_file = name + "_compression_block.v",
        visibility = ["//visibility:private"],
        target_compatible_with = ["//xls/build_rules:has_env_vars"], # Prevent this rule from being called in CI by `bazel test //xls/...`
    )

    verilog_library(
        name = name + "_compression_block_verilog_lib",
        srcs = [
            ":" + name + "_compression_block.v",
        ],
        visibility = ["//visibility:private"],
        target_compatible_with = ["//xls/build_rules:has_env_vars"], # Prevent this rule from being called in CI by `bazel test //xls/...`
    )

    synthesize_rtl(
        name = name + "_compression_block_synth",
        standard_cells = "@org_theopenroadproject_asap7//:asap7_rvt_1x",
        top_module = name,
        deps = [
            ":" + name + "_compression_block_verilog_lib",
        ],
        visibility = ["//visibility:private"],
        target_compatible_with = ["//xls/build_rules:has_env_vars"], # Prevent this rule from being called in CI by `bazel test //xls/...`
    )

    place_and_route(
        name = name + "_compression_block_place_and_route",
        clock_period = "$env(CLOCK_PERIOD)",
        core_padding_microns = 5,
        placement_density = "$env(PLACEMENT_DENSITY)",
        synthesized_rtl = ":" + name + "_compression_block_synth",
        target_die_utilization_percentage = "$env(TARGET_DIE_UTILIZATION_PERCENTAGE)",
        target_compatible_with = ["//xls/build_rules:has_env_vars"], # Prevent this rule from being called in CI by `bazel test //xls/...`
    )

    py_test(
        name = name + "_compression_block_cocotb_test",
        srcs = [cocotb_test],
        main = cocotb_test,
        data = [
            ":" + name + "_compression_block_verilog",
            "@com_icarus_iverilog//:iverilog",
            "@com_icarus_iverilog//:vvp",
        ],
        deps = [
            "//xls/common:runfiles",
            "//xls/common:test_base",
            "//xls/simulation/cocotb:cocotb_struct",
            "//xls/simulation/cocotb:cocotb_xls",
            "//xls/tools/dashboard:utils",
        ],
        imports = ["."],
        target_compatible_with = ["//xls/build_rules:has_env_vars"], # Prevent this rule from being called in CI by `bazel test //xls/...`
    )

    test_label = native.package_name() + ":" + name + "_compression_block_cocotb_test"
    vizier_optimize(
        name = name + "_compression_block_parameter_optimization",
        test = test_label,
        parser = test_parser,
        log_file = test_log_file,
        iterations = opt_eng_iterations,
        suggestions = opt_eng_suggestions,
        algorithm = opt_eng_algorithm,
    )

def _vizier_optimize_impl(ctx):
    executable = ctx.actions.declare_file("{}_run.sh".format(ctx.attr.name))
    command = "python {} --root-dir {} --test-label \"{}\" --parser {} --log-file {} --iterations {} --suggestions {} --algorithm {}".format(ctx.executable._run.short_path, "$BUILD_WORKSPACE_DIRECTORY", ctx.attr.test, ctx.executable.parser.path, ctx.attr.log_file, ctx.attr.iterations, ctx.attr.suggestions, ctx.attr.algorithm)

    ctx.actions.write(output = executable, content = command)

    return [
        DefaultInfo(
            executable = executable,
            runfiles = ctx.attr._run.default_runfiles.merge(ctx.attr.parser.default_runfiles),
        ),
    ]

vizier_optimize = rule(
    implementation = _vizier_optimize_impl,
    executable = True,
    attrs = {
        "test": attr.string(doc = "Label string for python script with cocotb test"),
        "parser": attr.label(cfg = "exec", executable = True, default = "//xls/tools/dashboard:generic_parser", doc = "Label for passing the parser for test output"),
        "log_file": attr.string(default = "/tmp/test_log.txt", doc = "path to temporary log file with test output"),
        "iterations": attr.int(default = 5, doc = "Max iteration count for the optimization engine"),
        "suggestions": attr.int(default = 1, doc = "Max suggestion count per iteration"),
        "algorithm": attr.string(default = "NSGA2", values = ["GAUSSIAN_PROCESS_BANDIT", "RANDOM_SEARCH", "QUASI_RANDOM_SEARCH", "GRID_SEARCH", "SHUFFLED_GRID_SEARCH", "EAGLE_STRATEGY", "NSGA2", "BOCS", "HARMONICA"], doc = "Algorithm for the optimization engine"),
        "_run": attr.label(
            cfg = "exec",
            executable = True,
            default = Label("//xls/tools:vizier_runner"),
        ),
    },
)
"""Runs the `vizier_runner` python script

This rule runs `vizier_runner` script which executes the vizier parameter optimization
for specified DSLX (de)compression block.
"""
