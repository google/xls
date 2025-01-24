# Copyright 2022 The XLS Authors
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

"""Convenience macro for DSLX language-level tests and associated targets.

Often when we define a DSLX language-level test we also want to know that file
converts to IR and evaluates as IR appropriately.

Note: these are not user-facing, these are just for convenience of internal
development in specifying the various targets we usually want stood up as part
of defining a DSLX language-level test.
"""

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
    "xls_dslx_fmt_test",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_eval_ir_test",
    "xls_ir_equivalence_test",
)

def dslx_lang_test(
        name,
        dslx_deps = None,
        dslx_entry = "main",
        ir_entry = None,
        convert_to_ir = True,
        test_ir_equivalence = True,
        evaluate_ir = True,
        benchmark_ir = True,
        warnings_as_errors = True,
        test_autofmt = True,
        compare = "jit"):
    """This macro is convenient shorthand for our many DSLX test targets.

    The primary target that it generates that developers may want to depend upon is:

      :${name}_dslx

    Which is the DSLX library created by this macro.

    It also, depending on given flags, can create several test targets:

    ```
    $ bazel query 'tests(//xls/dslx/tests:all)' | grep compound_eq
    //xls/dslx/tests:compound_eq_dslx_test
    //xls/dslx/tests:compound_eq_eval_ir_test
    //xls/dslx/tests:compound_eq_ir_equivalence_test
    ```

    And it also, depending on given flags, can create a non-test IR
    benchmarking target -- when run manually this summarizes the critical path
    of the resulting optimized IR (using the default delay model):

    ```
    $ bazel run -c opt //xls/dslx/tests:compound_eq_benchmark_ir
    ```

    Args:
      name: Base name for the targets, and identifies the primary DSLX source
        for the test (e.g. "{name}.x"). Notice there is no 'srcs' attribute --
        this is effectively it.
      dslx_deps: Dependency labels (for other xls_dslx_library targets required
        as deps of the library).
      dslx_entry: DSLX-level entry point name. By default, the value is "main".
      ir_entry: IR-level entry point name, if it is required to be explicitly
        specified (e.g. for instantiated parametric entry points). Note this
        would be a mangled DSLX name.
      convert_to_ir: Whether this target should also be converted to IR (for
        test files that are "tests only" there is only a trivial IR
        conversion, so we avoid it explicitly as things like "IR tests" will
        fail on empty IR conversions).
      test_ir_equivalence: Whether to enable IR equivalence (unopt vs opt)
        testing for the language-level test. Only used when convert_to_ir is
        true.
      evaluate_ir: Whether to enable IR evaluation testing for the
        language-level target. Only used when convert_to_ir is true.
      benchmark_ir: Whether to enable the IR benchmarking tool to run on the
        converted IR. Only used when convert_to_ir is true.
      warnings_as_errors: Whether warnings are errors within the DSLX library definition.
      test_autofmt: Whether to make an autoformatting test target. This ensures the
        language test remains auto-formatted.
      compare: Whether to compare DSL-interpreted results with IR execution for each
        function for consistency checking.

    As a byproduct this makes a "{name}_dslx" library target that other
    dslx_interp_tests can reference via the dslx_deps attribute.
    """
    if compare not in ["none", "jit", "interpreter"]:
        fail("compare must be one of: none, jit, interpreter")

    dslx_deps = dslx_deps or []
    xls_dslx_library(
        name = name + "_dslx",
        srcs = [name + ".x"],
        deps = dslx_deps,
        warnings_as_errors = warnings_as_errors,
    )

    test_args = {} if convert_to_ir else {"compare": compare}
    test_args["warnings_as_errors"] = "true" if warnings_as_errors else "false"
    xls_dslx_test(
        name = name + "_dslx_test",
        library = name + "_dslx",
        dslx_test_args = test_args,
    )

    if test_autofmt:
        xls_dslx_fmt_test(
            name = name + "_dslx_fmt",
            src = name + ".x",
            # We enable the opportunistic postcondition for all of our
            # auto-formatted language tests, because we don't generally expect
            # we've put non-canonical constructs in our language tests; e.g.
            # unnecessarily nested parens, unnecessary struct field
            # identifiers, etc.
            opportunistic_postcondition = True,
        )

    if convert_to_ir:
        # Note: this generates output files with both .ir and .opt.ir suffixes.
        xls_dslx_opt_ir(
            name = "_" + name + "_ir",
            srcs = [name + ".x"],
            deps = dslx_deps,
            dslx_top = dslx_entry,
            ir_conv_args = {
                "warnings_as_errors": test_args["warnings_as_errors"],
            },
        )
        if test_ir_equivalence:
            xls_ir_equivalence_test(
                name = name + "_ir_equivalence_test",
                src_0 = ":_" + name + "_ir.ir",
                src_1 = ":_" + name + "_ir.opt.ir",
                top = ir_entry,
            )
        if evaluate_ir:
            xls_eval_ir_test(
                name = name + "_eval_ir_test",
                src = ":_" + name + "_ir.opt.ir",
                top = ir_entry,
            )
        if benchmark_ir:
            xls_benchmark_ir(
                name = name + "_benchmark_ir",
                src = ":_" + name + "_ir.opt.ir",
                top = ir_entry,
            )
