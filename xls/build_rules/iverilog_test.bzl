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

load("@rules_shell//shell:sh_test.bzl", "sh_test")
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

    iverilog_tool = Label("@iverilog//:iverilog")
    ivl_base = Label("@iverilog//:ivl_base")
    vvp_tool = Label("@iverilog//:vvp")
    compiled = name + ".iverilog.out"
    native.genrule(
        name = name + "-iverilog-build",
        srcs = srcs,
        # Note: GENDIR is a builtin environment variable for genrule commands
        # that points at the genfiles' location, we have to add it to the
        # searched set of paths for inclusions so we can include generated
        # verilog files as we can generated C++ files in cc_library rules.
        # Icarus loads VPI modules during compilation and embeds their paths in
        # the output. Rewrite those paths so VVP finds them in the test runfiles.
        cmd = (
            'IVL_BASE="$(execpath %s)" "$(execpath %s)" ' % (ivl_base, iverilog_tool) +
            "-DIVERILOG -s %s $(execpath %s) %s -o $@.tmp -g2001 -I$(GENDIR) && " % (top, main, defines) +
            """sed 's|^:vpi_module "$(execpath %s)/|:vpi_module "$(rootpath %s)/|' "$@.tmp" > "$@" && """ % (ivl_base, ivl_base) +
            'rm "$@.tmp"'
        ),
        outs = [compiled],
        tools = [
            iverilog_tool,
            ivl_base,
        ],
    )
    if execute:
        sh_test(
            name = name + "-run_test",
            srcs = [Label("//xls/build_rules:vvp_test_runner.sh")],
            args = [
                "$(rootpath %s)" % vvp_tool,
                "$(rootpath %s)" % ivl_base,
                "$(rootpath :%s)" % compiled,
            ],
            data = [
                ":" + compiled,
                ivl_base,
                vvp_tool,
            ],
        )
        tests.append(":" + name + "-run_test")

    native.test_suite(
        name = name,
        tests = tests,
    )
