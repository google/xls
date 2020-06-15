# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains macros for DSLX test targets."""

# genrules are used in this file
load("@bazel_skylib//rules:build_test.bzl", "build_test")

_IR_CONVERTER_MAIN = "//xls/dslx:ir_converter_main"
_OPT_MAIN = "//xls/tools:opt_main"
_CODEGEN_MAIN = "//xls/tools:codegen_main"
_DSLX_TEST = "//xls/dslx/interpreter:dslx_test"
_INTERPRETER_MAIN = "//xls/dslx/interpreter:interpreter_main"

DEFAULT_DELAY_MODEL = "unit"

def _codegen_stem(codegen_params):
    """Returns a string based on codegen params for use in target names.

    String contains notable elements from ths codegen parameters such as clock
    period, delay model, etc.

    Args:
      codegen_params: Codegen parameters.

    Returns:
      String based on codegen params.
    """
    delay_model = codegen_params.get("delay_model", DEFAULT_DELAY_MODEL)
    if "clock_period_ps" in codegen_params:
        return "clock_{}ps_model_{}".format(
            codegen_params["clock_period_ps"],
            delay_model,
        )
    else:
        return "stages_{}_model_{}".format(
            codegen_params["pipeline_stages"],
            delay_model,
        )

def _codegen(
        name,
        srcs,
        codegen_params,
        entry = None,
        tags = []):
    """Generates a Verilog file by running codegen_main on the source IR files.

    Args:
      name: Name of the Verilog file to generate.
      srcs: IR sources.
      codegen_params: Codegen configuration used for Verilog generation.
      entry: Name of entry function to codegen.
      tags: Tags to add to RTL target.
    """
    codegen_flags = []
    codegen_flags.append("--delay_model=" +
                         codegen_params.get("delay_model", DEFAULT_DELAY_MODEL))

    CODEGEN_FLAGS = (
        "clock_period_ps",
        "pipeline_stages",
        "entry",
        "input_valid_signal",
        "output_valid_signal",
        "module_name",
        "clock_margin_percent",
        "flop_inputs",
        "flop_outputs",
    )
    for flag_name in CODEGEN_FLAGS:
        if flag_name in codegen_params:
            codegen_flags.append("--{}={}".format(
                flag_name,
                codegen_params[flag_name],
            ))
    verilog_file = name + ".v"
    module_sig_file = name + ".sig.pbtxt"
    schedule_file = name + ".schedule.textproto"
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [verilog_file, module_sig_file, schedule_file],
        cmd = ("$(location %s) %s --output_signature_path=$(@D)/%s " +
               "--output_verilog_path=$(@D)/%s " +
               "--output_schedule_path=$(@D)/%s $<") % (
            _CODEGEN_MAIN,
            " ".join(codegen_flags),
            module_sig_file,
            verilog_file,
            schedule_file,
        ),
        exec_tools = [_CODEGEN_MAIN],
        tags = tags,
    )

def _make_benchmark_args(package_name, name, entry, args):
    benchmark_args = [package_name + "/" + name + ".ir"]
    if entry:
        benchmark_args.append("--entry={}".format(entry))
    benchmark_args += args
    return benchmark_args

def dslx_codegen(name, dslx_dep, configs, entry = None, tags = None):
    """Exercises code generation to create Verilog (post IR conversion).

    Multiple code generation configurations can be given.

    Args:
      name: Describes base name of the targets to create; must be suffixed with
        "_codegen".
      dslx_dep: A label that indicates where the IR targets live;
        that is, it is the corresponding dslx_test rule's "name" as a label.
      configs: List of code-generation configurations, which can specify
        any/all of: clock_period_ps, pipeline_stages, entry,
        clock_margin_percent, delay_model.
      entry: Entry function name to use for code generation.
      tags: Tags to use for the resulting test targets.
    """
    if not name.endswith("_codegen"):
        fail("Codegen name must end with '_codegen': " + repr(name))
    base_name = name[:-len("_codegen")]
    tags = tags or []
    package_name = dslx_dep.split(":")[0].lstrip("/") or native.package_name()
    for params in configs:
        _codegen(
            name = "{}_{}".format(base_name, _codegen_stem(params)),
            srcs = [dslx_dep + "_opt_ir"],
            codegen_params = params,
            entry = entry,
            tags = tags,
        )

        # Also create a codegen benchmark target.
        codegen_benchmark_args = _make_benchmark_args(package_name, base_name + ".opt", entry, args = [])
        codegen_benchmark_args.append("--delay_model={}".format(
            params.get("delay_model", DEFAULT_DELAY_MODEL),
        ))
        for flag_name in (
            "clock_period_ps",
            "pipeline_stages",
            "entry",
            "clock_margin_percent",
        ):
            if flag_name in params:
                codegen_benchmark_args.append("--{}={}".format(
                    flag_name,
                    params[flag_name],
                ))

        native.sh_test(
            name = "{}_benchmark_codegen_test_{}".format(
                base_name,
                _codegen_stem(params),
            ),
            srcs = ["//xls/tools:benchmark_test_sh"],
            args = codegen_benchmark_args,
            data = [
                "//xls/dslx:ir_converter_main",
                "//xls/tools:benchmark_main",
                "//xls/tools:opt_main",
                dslx_dep + "_all_ir",
            ],
            tags = tags,
        )

# TODO(meheff): dslx_test includes a bunch of XLS internal specific stuff such
# as generating benchmarks and convert IR. These should be factored out so we
# have a clean macro for end-user use.
def dslx_test(
        name,
        srcs,
        deps = None,
        entry = None,
        args = None,
        convert_ir = True,
        prove_unopt_eq_opt = True,
        generate_benchmark = True,
        tags = []):
    """Runs all test cases inside of a DSLX source file as a test target.

    Args:
      name: 'Base' name for the targets that get created.
      srcs: '.x' file sources.
      deps: Dependent '.x' file sources.
      entry: Name (currently *mangled* name) of the entry point that should be
        converted / code generated.
      args: Additional arguments to pass to the DSLX interpreter and IR
        converter.
      convert_ir: Whether or not to convert the DSLX code to IR.
      generate_benchmark: Whether or not to create a benchmark target (that
        analyses XLS scheduled critical path).
      prove_unopt_eq_opt: Whether or not to generate a test to compare semantics
        of opt vs. non-opt IR. Only enabled if convert_ir is true.
      tags: Tags to place on all generated targets.
    """
    args = args or []
    deps = deps or []
    if len(srcs) != 1:
        fail("More than one source not currently supported.")
    if entry and not type(entry) != str:
        fail("Entry argument must be a string.")
    src = srcs[0]

    native.sh_test(
        name = name + "_dslx_test",
        srcs = [_DSLX_TEST],
        args = [native.package_name() + "/" + src] + args,
        data = [
            _INTERPRETER_MAIN,
        ] + srcs + deps,
        tags = tags,
    )

    # TODO(meheff): Move this to a different internal-only bzl file.
    if convert_ir:
        native.sh_test(
            name = name + "_ir_converter_test",
            srcs = ["//xls/dslx:ir_converter_test_sh"],
            args = [native.package_name() + "/" + src] + args,
            data = [
                "//xls/dslx:ir_converter_main",
            ] + srcs + deps,
            tags = tags,
        )
        native.genrule(
            name = name + "_ir",
            srcs = srcs + deps,
            outs = [name + ".ir"],
            cmd = "$(location //xls/dslx:ir_converter_main) $(SRCS) > $@",
            exec_tools = ["//xls/dslx:ir_converter_main"],
            tags = tags,
        )
        native.genrule(
            name = name + "_opt_ir",
            srcs = srcs + deps,
            outs = [name + ".opt.ir"],
            cmd = "$(location //xls/dslx:ir_converter_main) $(SRCS) | $(location //xls/tools:opt_main) --entry=%s - > $@" % (entry or ""),
            exec_tools = [
                "//xls/dslx:ir_converter_main",
                "//xls/tools:opt_main",
            ],
            tags = tags,
        )
        native.filegroup(
            name = name + "_all_ir",
            srcs = [name + ".opt.ir", name + ".ir"],
        )

        if prove_unopt_eq_opt:
            native.sh_test(
                name = name + "_opt_equivalence_test",
                srcs = ["//xls/tools:check_ir_equivalence_sh"],
                args = [
                    native.package_name() + "/" + name + ".ir",
                    native.package_name() + "/" + name + ".opt.ir",
                ] + (["--function=" + entry] if entry else []),
                size = "large",
                data = [
                    ":" + name + "_all_ir",
                    "//xls/tools:check_ir_equivalence_main",
                ],
                #tags = tags + ["manual", "optonly"],
                tags = tags + ["optonly"],
            )

        if generate_benchmark:
            benchmark_args = _make_benchmark_args(native.package_name(), name, entry, args)

            # Add test which executes benchmark_main on the IR.
            native.sh_test(
                name = name + "_benchmark_test",
                srcs = ["//xls/tools:benchmark_test_sh"],
                args = benchmark_args,
                data = [
                    "//xls/tools:benchmark_main",
                    ":" + name + "_all_ir",
                ],
                tags = tags,
            )

            # Add test which evaluates the IR with the interpreter and verifies
            # the result before and after optimizations match.
            native.sh_test(
                name = name + "_benchmark_eval_test",
                srcs = ["//xls/tools:benchmark_eval_test_sh"],
                args = benchmark_args + ["--random_inputs=100", "--optimize_ir"],
                data = [
                    "//xls/tools:eval_ir_main",
                    ":" + name + "_all_ir",
                ],
                tags = tags + ["optonly"],
            )

    native.filegroup(
        name = name + "_source",
        srcs = srcs,
    )
    native.test_suite(
        name = name,
        tests = [name + "_dslx_test"],
        tags = tags,
    )

# TODO(meheff): This macro should define some tests to sanity check the
# generated RTL.
def dslx_generated_rtl(
        name,
        srcs,
        codegen_params,
        deps = None,
        entry = None,
        tags = []):
    """Generates RTL from DSLX sources using a released toolchain.

    Args:
      name: Base name for the targets that get created. The Verilog file will
        have the name '{name}.v'.
      srcs: '.x' file sources.
      codegen_params: Codegen configuration used for Verilog generation.
      deps: Dependent '.x' file sources.
      entry: Name of entry function to codegen.
      tags: Tags to place on all generated targets.
    """
    deps = deps or []
    if len(srcs) != 1:
        fail("More than one source not currently supported.")
    src = srcs[0]

    native.genrule(
        name = name + "_ir",
        srcs = [src] + deps,
        outs = [name + ".ir"],
        cmd = "$(location %s) $(SRCS) > $@" % _IR_CONVERTER_MAIN,
        exec_tools = [_IR_CONVERTER_MAIN],
        tags = tags,
    )

    native.genrule(
        name = name + "_opt_ir",
        srcs = [":{}_ir".format(name)],
        outs = [name + ".opt.ir"],
        cmd = "$(location %s) --entry=%s  $(SRCS) > $@" % (_OPT_MAIN, entry or ""),
        exec_tools = [_OPT_MAIN],
        tags = tags,
    )

    _codegen(
        name,
        srcs = [name + "_opt_ir"],
        codegen_params = codegen_params,
        entry = entry,
        tags = tags,
    )

    # Add a build test to ensure changes to BUILD and bzl files do not break
    # targets built with released toolchains.
    build_test(
        name = name + "_build_test",
        targets = [":" + name],
    )
