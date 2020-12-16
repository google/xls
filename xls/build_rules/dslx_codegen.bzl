# Copyright 2020 The XLS Authors
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

"""See dslx_codegen()."""

load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")

DEFAULT_DELAY_MODEL = "unit"
_CODEGEN_MAIN = "//xls/tools:codegen_main"

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

def codegen(
        name,
        srcs,
        codegen_params,
        entry = None,
        tags = [],
        **kwargs):
    """Generates a Verilog file by running codegen_main on the source IR files.

    Args:
      name: Name of the Verilog file to generate.
      srcs: IR sources.
      codegen_params: Codegen configuration used for Verilog generation.
      entry: Name of entry function to codegen.
      tags: Tags to add to RTL target.
      **kwargs: Extra arguments to pass to genrule.
    """
    codegen_flags = []
    codegen_flags.append("--delay_model=" +
                         codegen_params.get("delay_model", DEFAULT_DELAY_MODEL))

    CODEGEN_FLAGS = (
        "clock_margin_percent",
        "clock_period_ps",
        "entry",
        "flop_inputs",
        "flop_outputs",
        "input_valid_signal",
        "module_name",
        "output_valid_signal",
        "pipeline_stages",
        "reset",
        "reset_active_low",
        "reset_asynchronous",
    )
    for flag_name in CODEGEN_FLAGS:
        if flag_name in codegen_params:
            codegen_flags.append("--{}={}".format(
                flag_name,
                codegen_params[flag_name],
            ))
    verilog_file = name + ".v"
    module_sig_file = name + ".sig.textproto"
    schedule_file = name + ".schedule.textproto"
    genrule_wrapper(
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
        **kwargs
    )

def make_benchmark_args(package_name, name, entry, args):
    """Helper for creating benchmark executable args.

    Args:
      package_name: Prefix to use for the IR path.
      name: File name to use in IR path.
      entry: Entry point to use in XLS IR file for benchmarking.
      args: Any additional args to pass.

    Returns:
      List of arguments to use in benchmark executable.
    """
    benchmark_args = [package_name + "/" + name + ".ir"]
    if entry:
        benchmark_args.append("--entry={}".format(entry))
    benchmark_args += args
    return benchmark_args

def dslx_codegen(
        name,
        dslx_dep,
        configs,
        entry = None,
        tags = None,
        **kwargs):
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
      **kwargs: Extra arguments to pass to code generation (`codegen`).
    """
    if not name.endswith("_codegen"):
        fail("Codegen name must end with '_codegen': " + repr(name))
    base_name = name[:-len("_codegen")]
    tags = tags or []
    package_name = dslx_dep.split(":")[0].lstrip("/") or native.package_name()
    for params in configs:
        codegen(
            name = "{}_{}".format(base_name, _codegen_stem(params)),
            srcs = [dslx_dep + "_opt_ir"],
            codegen_params = params,
            entry = entry,
            tags = tags,
            **kwargs
        )

        # Also create a codegen benchmark target.
        codegen_benchmark_args = make_benchmark_args(package_name, dslx_dep.lstrip(":") + ".opt", entry, args = [])
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
