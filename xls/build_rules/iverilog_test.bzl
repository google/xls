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

"""Contains internal XLS macros."""

load("//xls/build_rules:elab_test.bzl", "elab_test")

def iverilog_test(name, top, main, srcs, execute = True, tick_defines = None):
    """Defines Icarus Verilog test, with associated elaboration test.

    The elaboration test helps to identify issues where iverilog is overly
    lenient.

    Args:
      name: Name of the target. This name will be a suite of the elaboration test
        (-elab_test suffix) and iverilog simulation test (-run_test suffix).
      top: Top-level module name for the test to run.
      main: Main source file to compile via iverilog.
      srcs: Supporting source files, including ``main``.
      execute: If true then run the test through Iverilog. Otherwise only linting
        and elaboration is performed.
      tick_defines: A map containing Verilog tick-defines
        (eg, "`define FOO 0").
    """
    if main not in srcs:
        fail("main verilog source %r not in srcs %r" % (main, srcs))

    tick_defines = tick_defines or {}

    tests = []

    et = elab_test(
        name,
        src = main,
        hdrs = [src for src in srcs if src != main],
        top = top,
    )
    if et:
        tests.append(et)

    defines = " ".join(
        ["-D{}={}".format(k, v) for (k, v) in sorted(tick_defines.items())],
    )

    native.genrule(
        name = name + "-iverilog-build",
        srcs = srcs,
        # Note: GENDIR is a builtin environment variable for genrule commands
        # that points at the genfiles' location, we have to add it to the
        # searched set of paths for inclusions so we can include generated
        # verilog files as we can generated C++ files in cc_library rules.
        cmd = "$(location @com_icarus_iverilog//:iverilog) -s %s $(location %s) %s -o $@ -g2001 -I$(GENDIR)" % (top, main, defines),
        outs = [name + ".iverilog.out"],
        tools = [
            "@com_icarus_iverilog//:iverilog",
            "@com_icarus_iverilog//:vvp",
        ],
    )
    if execute:
        native.genrule(
            name = name + "-vvp-runner",
            srcs = [":" + name + "-iverilog-build"],
            cmd = "$(location //xls/dev_tools:generate_vvp_runner) $< > $@",
            outs = [name + "-vvp-runner.sh"],
            tools = ["//xls/dev_tools:generate_vvp_runner"],
        )
        native.sh_test(
            name = name + "-run_test",
            srcs = [":" + name + "-vvp-runner"],
            data = [
                ":" + name + "-iverilog-build",
                "@com_icarus_iverilog//:vvp",
            ],
        )
        tests.append(":" + name + "-run_test")

    native.test_suite(
        name = name,
        tests = tests,
    )
