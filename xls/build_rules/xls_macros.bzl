# Copyright 2021 The XLS Authors
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

"""
This module contains build macros for XLS.
"""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:diff_test.bzl", "diff_test")
load("@rules_hdl//synthesis:build_defs.bzl", "benchmark_synth", "synthesize_rtl")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "append_xls_ir_verilog_generated_files",
    "get_xls_ir_verilog_generated_files",
    "validate_verilog_filename",
    "xls_ir_verilog",
)
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "split_filename",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL",
    "delay_model_to_standard_cells",
    "enable_generated_file_wrapper",
)
load(
    "//xls/build_rules:xls_dslx_rules.bzl",
    "xls_dslx_generate_cpp_type_files",
)
load(
    "//xls/build_rules:xls_ir_macros.bzl",
    _xls_ir_opt_ir_macro = "xls_ir_opt_ir_macro",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "append_xls_dslx_ir_generated_files",
    "append_xls_ir_opt_ir_generated_files",
    "get_xls_dslx_ir_generated_files",
    "get_xls_ir_opt_ir_generated_files",
    "xls_benchmark_ir",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    "xls_dslx_opt_ir",
    "xls_dslx_verilog",
)
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "DEFAULT_DSLX_FMT_TARGET",
)
load(
    "//xls/build_rules:xls_type_check_utils.bzl",
    "bool_type_check",
    "dictionary_type_check",
    "list_type_check",
    "string_type_check",
)

def _xls_dslx_verilog_macro(
        name,
        dslx_top,
        verilog_file,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from a DSLX source file.

    The macro instantiates a build rule that generates a Verilog file from a DSLX source file. The
    build rule executes the core functionality of following macros:

    1. xls_dslx_ir (converts a DSLX file to an IR),
    1. xls_ir_opt_ir (optimizes the IR), and,
    1. xls_ir_verilog (generated a Verilog file).

    The macro also instantiates the 'enable_generated_file_wrapper'
    function. The generated files are listed in the outs attribute of the rule.

    This macro is used by the 'xls_dslx_verilog_build_and_test' macro.

    Examples:

    1. A simple example.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_verilog(
            name = "d_verilog",
            srcs = ["d.x"],
            deps = [":bc_dslx"],
            codegen_args = {
                "pipeline_stages": "1",
            },
            dslx_top = "d",
        )
        ```

    Args:
      name: The name of the rule.
      srcs: Top level source files for the conversion. Files must have a '.x'
        extension. There must be single source file.
      deps: Dependency targets for the files in the 'srcs' argument.
      library: A DSLX library target where the direct (non-transitive)
        files of the target are tested. This argument is mutually
        exclusive with the 'srcs' and 'deps' arguments.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' extension.
      dslx_top: The top entity to perform the IR conversion.
      ir_conv_args: Arguments of the IR conversion tool. For details on the
        arguments, refer to the ir_converter_main application at
        //xls/dslx/ir_convert/ir_converter_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      codegen_args: Arguments of the codegen tool. For details on the arguments,
        refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    list_type_check("srcs", srcs, True)
    list_type_check("deps", deps, True)
    string_type_check("library", library, True)
    string_type_check("dslx_top", dslx_top)
    string_type_check("verilog_file", verilog_file)
    dictionary_type_check("ir_conv_args", ir_conv_args)
    dictionary_type_check("opt_ir_args", opt_ir_args)
    dictionary_type_check("codegen_args", codegen_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = append_xls_dslx_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)
    use_system_verilog = codegen_args.get("use_system_verilog", "True").lower() == "true"
    validate_verilog_filename(verilog_file, use_system_verilog)
    verilog_basename = split_filename(verilog_file)[0]
    kwargs = append_xls_ir_verilog_generated_files(
        kwargs,
        verilog_basename,
        codegen_args,
    )

    xls_dslx_verilog(
        name = name,
        srcs = srcs,
        deps = deps,
        library = library,
        dslx_top = dslx_top,
        verilog_file = verilog_file,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        outs = get_xls_dslx_ir_generated_files(kwargs) +
               get_xls_ir_opt_ir_generated_files(kwargs) +
               get_xls_ir_verilog_generated_files(kwargs, codegen_args) +
               [verilog_file],
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_dslx_verilog_build_and_test(
        name,
        dslx_top,
        verilog_file,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from a DSLX source file and tests the build.

    The macro instantiates a build rule that generates a Verilog file from a DSLX source file. The
    build rule executes the core functionality of following macros:

    1. xls_dslx_ir (converts a DSLX file to an IR),
    1. xls_ir_opt_ir (optimizes the IR), and,
    1. xls_ir_verilog (generated a Verilog file).

    The macro also instantiates a 'build_test' testing that the build rule generating a Verilog
    file. If the build is not successful, an error is produced when executing a test command on the
    target.

    Examples:

    1. A simple example.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_verilog(
            name = "d_verilog",
            srcs = ["d.x"],
            deps = [":bc_dslx"],
            codegen_args = {
                "pipeline_stages": "1",
            },
            dslx_top = "d",
        )
        ```

    Args:
      name: The name of the rule.
      srcs: Top level source files for the conversion. Files must have a '.x'
        extension. There must be single source file.
      deps: Dependency targets for the files in the 'srcs' argument.
      library: A DSLX library target where the direct (non-transitive)
        files of the target are tested. This argument is mutually
        exclusive with the 'srcs' and 'deps' arguments.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' extension.
      dslx_top: The top entity to perform the IR conversion.
      ir_conv_args: Arguments of the IR conversion tool. For details on the
        arguments, refer to the ir_converter_main application at
        //xls/dslx/ir_convert/ir_converter_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      codegen_args: Arguments of the codegen tool. For details on the arguments,
        refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """
    _xls_dslx_verilog_macro(
        name = name,
        dslx_top = dslx_top,
        verilog_file = verilog_file,
        srcs = srcs,
        deps = deps,
        library = library,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )
    build_test(
        name = "__" + name,
        targets = [":" + name],
        tags = kwargs["tags"] if "tags" in kwargs else [],
    )

def xls_dslx_opt_ir_macro(
        name,
        dslx_top,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        opt_ir_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating an optimized IR file from a DSLX source file.

    The macro instantiates a build rule that generates an optimized IR file from
    a DSLX source file. The build rule executes the core functionality of
    following macros:

    1. xls_dslx_ir (converts a DSLX file to an IR), and,
    1. xls_ir_opt_ir (optimizes the IR).

    The macro also instantiates the 'enable_generated_file_wrapper'
    function. The generated files are listed in the outs attribute of the rule.

    Examples:

    1. A simple example.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_opt_ir(
            name = "d_opt_ir",
            srcs = ["d.x"],
            deps = [":bc_dslx"],
            dslx_top = "d",
        )
        ```

    Args:
      name: The name of the rule.
      srcs: Top level source files for the conversion. Files must have a '.x'
        extension. There must be single source file.
      deps: Dependency targets for the files in the 'srcs' argument.
      library: A DSLX library target where the direct (non-transitive)
        files of the target are tested. This argument is mutually
        exclusive with the 'srcs' and 'deps' arguments.
      dslx_top: The top entity to perform the IR conversion.
      ir_conv_args: Arguments of the IR conversion tool. For details on the
        arguments, refer to the ir_converter_main application at
        //xls/dslx/ir_convert/ir_converter_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    list_type_check("srcs", srcs, True)
    list_type_check("deps", deps, True)
    string_type_check("library", library, True)
    string_type_check("dslx_top", dslx_top)
    dictionary_type_check("ir_conv_args", ir_conv_args)
    dictionary_type_check("opt_ir_args", opt_ir_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = append_xls_dslx_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)

    xls_dslx_opt_ir(
        name = name,
        srcs = srcs,
        deps = deps,
        library = library,
        dslx_top = dslx_top,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        outs = get_xls_dslx_ir_generated_files(kwargs) +
               get_xls_ir_opt_ir_generated_files(kwargs),
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_dslx_cpp_type_library(
        name,
        src,
        deps = [],
        namespace = None):
    """Creates a cc_library target for transpiled DSLX types.

    This macros invokes the DSLX-to-C++ transpiler and compiles the result as
    a cc_library with its target name identical to this macro.

    Args:
      name: The name of the eventual cc_library.
      src: The DSLX file whose types to compile as C++.
      namespace: The C++ namespace to generate the code in (e.g., `foo::bar`).
    """
    xls_dslx_generate_cpp_type_files(
        name = name + "_generate_sources",
        srcs = [src],
        source_file = name + ".cc",
        header_file = name + ".h",
        deps = deps,
        namespace = namespace,
    )

    native.cc_library(
        name = name,
        srcs = [":" + name + ".cc"],
        hdrs = [":" + name + ".h"],
        deps = [
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/status:status",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/types:span",
            "//xls/public:status_macros",
            "//xls/public:value",
        ],
        data = [
            ":" + name + "_generate_sources",
        ],
    )

def xls_synthesis_metrics(
        name,
        srcs,
        **kwargs):
    """Gather per-pipeline-stage metrics from log files.

    Gather per-stage post-synth metrics from the provided logs
    (from Yosys or OpenSTA) and save them in a "DesignStats" textproto.
    Recognized metrics from Yosys log:
      Total cell area (um^2).
      Logic levels
      Cell count
      Flop count
    Recognized metrics from OpenSTA log:
      Critical path delay (ps)
      Critical path start point
      Critical path end point

    Args:
        name: Output "DesignStats" textproto will be `<name>.textproto`
        srcs: Targets from which log files will be scanned.
              For post-synth, use "synthesize_rtl" and "run_opensta" targets.
        **kwargs: Accepts add'l keyword arguments. Passed to native.genrule().
    """
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [name + ".textproto"],
        cmd = "$(location //xls/tools:gather_design_stats) " +
              "--out $@ " +
              " ".join(["$(locations {}) ".format(s) for s in srcs]),
        tools = ["//xls/tools:gather_design_stats"],
        **kwargs
    )

def xls_dslx_fmt_test_macro(name, src, opportunistic_postcondition = False):
    """Creates a test target that confirms `src` is auto-formatted.

    Args:
        name: Name of the (diff) test target this will emit.
        src: Source file to auto-format.
        opportunistic_postcondition: Flag that checks whether the output text
            is highly similar to the input text. Note that sometimes this /can/
            flag an error for some set of valid auto-formattings, so is
            intended primarily for use as a development/debugging tool.
    """
    out = name + ".fmt.x"
    flag = "--opportunistic_postcondition" if opportunistic_postcondition else ""
    cmd = "$(location %s) %s $< > $@" % (Label(DEFAULT_DSLX_FMT_TARGET), flag)
    native.genrule(
        name = name + "_dslx_fmt",
        srcs = [src],
        outs = [out],
        tools = [Label(DEFAULT_DSLX_FMT_TARGET)],
        cmd = cmd,
    )

    # TODO(tedhong): 2023-11-02, adjust package_name depending on the WORKSPACE
    # the target is in.
    package_name = native.package_name()
    target = package_name + ":" + out
    src_file = package_name + "/" + src
    out_file = package_name + "/" + out
    diff_test(
        name = name,
        file1 = src,
        file2 = ":" + name + "_dslx_fmt",
        failure_message = "File %s was not canonically auto-formatted; to update, in the top level directory of your WORKSPACE run: bazel build -c opt %s && cp bazel-genfiles/%s %s" % (src_file, target, out_file, src_file),
    )

def xls_full_benchmark_ir_macro(
        name,
        src,
        synthesize = True,
        codegen_args = {},
        benchmark_ir_args = {},
        standard_cells = None,
        tags = None,
        ir_tags = None,
        synth_tags = None,
        **kwargs):
    """Executes the benchmark tool on an IR file.

Examples:

1. A file as the source.

    ```
    xls_benchmark_ir(
        name = "a_benchmark",
        src = "a.ir",
    )
    ```

1. An xls_ir_opt_ir target as the source.

    ```
    xls_ir_opt_ir(
        name = "a_opt_ir",
        src = "a.ir",
    )


    xls_benchmark_ir(
        name = "a_benchmark",
        src = ":a_opt_ir",
    )
    ```

    Args:
        name: A unique name for this target.
        src: The IR source file for the rule. A single source file must be provided. The file must
          have a '.ir' extension.
        synthesize: Add a synthesis benchmark in addition to the IR benchmark.
        codegen_args: Arguments of the codegen tool. For details on the arguments,
          refer to the codegen_main application at
          //xls/tools/codegen_main.cc.
        benchmark_ir_args: Arguments of the benchmark IR tool. For details on the arguments, refer
          to the benchmark_main application at //xls/dev_tools/benchmark_main.cc.
        standard_cells: Label for the PDK (possibly specifying a
          non-default corner), with the assumption that $location will
          return the timing (Liberty) library for the PDK corner. Unused if synthesize == False.
        tags: Tags for IR and synthesis benchmark targets.
        ir_tags: Tags for the IR benchmark target only.
        synth_tags: Tags for the synthesis and synthesis benchmark targets. Unused if synthesize == False.
        **kwargs: Keyword arguments for the IR benchmark target only.
    """

    list_type_check("tags", tags, can_be_none = True)
    list_type_check("ir_tags", ir_tags, can_be_none = True)
    list_type_check("synth_tags", synth_tags, can_be_none = True)
    tags = tags if tags != None else []
    ir_tags = ir_tags if ir_tags != None else []
    synth_tags = synth_tags if synth_tags != None else []

    # Setup shared arguments
    SHARED_FLAGS = (
        "top",
    )
    IR_OPT_FLAGS = (
        "ir_dump_path",
        "passes",
        "skip_passes",
        "opt_level",
        "convert_array_index_to_select",
        "inline_procs",
        "use_context_narrowing_analysis",
        "optimize_for_best_case_throughput",
    )
    opt_ir_args = {
        k: v
        for k, v in benchmark_ir_args.items()
        if k in IR_OPT_FLAGS or k in SHARED_FLAGS
    }
    benchmark_ir_codegen_args = {
        k: v
        for k, v in benchmark_ir_args.items()
        if k not in IR_OPT_FLAGS or k in SHARED_FLAGS
    }

    # Add default opt args (setting inline_procs to true by default so both branches are comparable)
    full_opt_args = {
        "inline_procs": "true",
    }
    full_opt_args.update(opt_ir_args)

    # Add default codegen args
    full_codegen_args = {
        "delay_model": DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL,
        "generator": "pipeline",
        "pipeline_stages": "1",
        "reset": "rst",
        "reset_data_path": "false",
        "use_system_verilog": "false",
        "module_name": name + "_default",
    }
    full_codegen_args.update(benchmark_ir_codegen_args)
    full_codegen_args.update(codegen_args)
    if "clock_period_ps" in full_codegen_args:
        full_codegen_args.pop("pipeline_stages")
    codegen_args = full_codegen_args

    delay_model = codegen_args["delay_model"]

    full_benchmark_ir_args = dict(full_codegen_args)
    full_benchmark_ir_args.update(benchmark_ir_args)

    if standard_cells == None:
        # Use default standard cells for the given delay model.
        standard_cells = delay_model_to_standard_cells(delay_model)

    # Some synthesis benchmarks require special permissions and amy not build so
    # mark these as manual.
    synth_tags.append("manual")

    # Create required targets.
    xls_benchmark_ir(
        name = name,
        src = src,
        benchmark_ir_args = full_benchmark_ir_args,
        tags = ir_tags + tags,
        **kwargs
    )
    if not synthesize:
        return

    opt_ir_target = name + ".default.opt_ir"
    _xls_ir_opt_ir_macro(
        name = opt_ir_target,
        src = src,
        opt_ir_args = full_opt_args,
        tags = tags,
    )

    codegen_target = "{}.default_{}.codegen".format(name, delay_model)
    verilog_file = codegen_target + ".v"
    xls_ir_verilog(
        name = codegen_target,
        src = ":{}".format(opt_ir_target),
        codegen_args = codegen_args,
        verilog_file = verilog_file,
        tags = tags,
    )
    verilog_target = "{}.default_{}.verilog".format(name, delay_model)
    verilog_library(
        name = verilog_target,
        srcs = [
            ":" + verilog_file,
        ],
        tags = tags,
    )
    synth_target = "{}.default_{}.synth".format(name, delay_model)
    synthesize_rtl(
        name = synth_target,
        standard_cells = standard_cells,
        top_module = codegen_args["module_name"],
        deps = [
            ":" + verilog_target,
        ],
        tags = synth_tags + tags,
    )
    benchmark_synth(
        name = "{}.default_{}.benchmark_synth".format(name, delay_model),
        synth_target = ":" + synth_target,
        tags = synth_tags + tags,
    )
