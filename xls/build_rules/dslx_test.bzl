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

"""See dslx_test()."""

load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")
load("//xls/build_rules:dslx_codegen.bzl", "make_benchmark_args")

_INTERPRETER_MAIN = "//xls/dslx:interpreter_main"
_DSLX_TEST = "//xls/dslx/interpreter:dslx_test"

# TODO(meheff): Move this to a different internal-only bzl file.
def _convert_ir(
        name,
        src,
        entry,
        srcs,
        deps,
        tags,
        args,
        prove_unopt_eq_opt,
        generate_benchmark,
        kwargs):
    native.sh_test(
        name = name + "_ir_converter_test",
        srcs = ["//xls/dslx:ir_converter_test_sh"],
        args = [native.package_name() + "/" + src] + args,
        data = [
            "//xls/dslx:ir_converter_main",
        ] + srcs + deps,
        tags = tags,
    )
    genrule_wrapper(
        name = name + "_ir",
        srcs = srcs + deps,
        outs = [name + ".ir"],
        cmd = "$(location //xls/dslx:ir_converter_main) --dslx_path=$(GENDIR) $(SRCS) > $@",
        exec_tools = ["//xls/dslx:ir_converter_main"],
        tags = tags,
        **kwargs
    )
    genrule_wrapper(
        name = name + "_opt_ir",
        srcs = srcs + deps,
        outs = [name + ".opt.ir"],
        cmd = ("$(location //xls/dslx:ir_converter_main) --dslx_path=$(GENDIR) $(SRCS) " +
               " | $(location //xls/tools:opt_main) --entry=%s - " +
               " > $@") % (entry or ""),
        exec_tools = [
            "//xls/dslx:ir_converter_main",
            "//xls/tools:opt_main",
        ],
        tags = tags,
        **kwargs
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
            tags = tags + ["optonly"],
        )

    if generate_benchmark:
        benchmark_args = make_benchmark_args(
            native.package_name(),
            name,
            entry,
            args,
        )

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
        compare = "jit",
        prove_unopt_eq_opt = True,
        generate_benchmark = True,
        tags = [],
        **kwargs):
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
      compare: Perform a runtime equivalence check between the DSLX interpreter
        and the IR JIT ('jit') or IR interpreter ('interpreter') or no IR
        conversion / comparison at all ('none').
      generate_benchmark: Whether or not to create a benchmark target (that
        analyses XLS scheduled critical path).
      prove_unopt_eq_opt: Whether or not to generate a test to compare semantics
        of opt vs. non-opt IR. Only enabled if convert_ir is true.
      tags: Tags to place on all generated targets.
      **kwargs: Extra arguments to pass to genrule.
    """
    args = args or []
    deps = deps or []
    if len(srcs) != 1:
        fail("More than one source not currently supported.")
    if entry and not type(entry) != str:
        fail("Entry argument must be a string.")
    src = srcs[0]

    interpreter_args = ["--compare={}".format(compare if convert_ir else "none")]
    native.sh_test(
        name = name + "_dslx_test",
        srcs = [_DSLX_TEST],
        args = [native.package_name() + "/" + src] + args + interpreter_args,
        data = [
            _INTERPRETER_MAIN,
        ] + srcs + deps,
        tags = tags,
    )

    if convert_ir:
        _convert_ir(
            name,
            src,
            entry,
            srcs,
            deps,
            tags,
            args,
            prove_unopt_eq_opt,
            generate_benchmark,
            kwargs,
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
