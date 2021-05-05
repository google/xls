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

"""See dslx_generated_rtl()."""

load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xls/build_rules:dslx_codegen.bzl", "codegen")

_IR_CONVERTER_MAIN = "//xls/dslx:ir_converter_main"
_OPT_MAIN = "//xls/tools:opt_main"

# TODO(meheff): This macro should define some tests to validate the generated
# RTL.
def dslx_generated_rtl(
        name,
        srcs,
        codegen_params,
        deps = None,
        entry = None,
        tags = [],
        **kwargs):
    """Generates RTL from DSLX sources using a released toolchain.

    Args:
      name: Base name for the targets that get created. The Verilog file will
        have the name '{name}.v'.
      srcs: '.x' file sources.
      codegen_params: Codegen configuration used for Verilog generation.
      deps: Dependent '.x' file sources.
      entry: Name of entry function to codegen.
      tags: Tags to place on all generated targets.
      **kwargs: Extra arguments to pass to genrule and _codegen.
    """
    deps = deps or []
    if len(srcs) != 1:
        fail("More than one source not currently supported.")
    src = srcs[0]

    genrule_wrapper(
        name = name + "_ir",
        srcs = [src] + deps,
        outs = [name + ".ir"],
        cmd = "$(location %s) --dslx_path=$(GENDIR) $(SRCS) > $@" % _IR_CONVERTER_MAIN,
        exec_tools = [_IR_CONVERTER_MAIN],
        tags = tags,
        **kwargs
    )

    genrule_wrapper(
        name = name + "_opt_ir",
        srcs = [":{}_ir".format(name)],
        outs = [name + ".opt.ir"],
        cmd = "$(location %s) --entry=%s  $(SRCS) > $@" % (_OPT_MAIN, entry or ""),
        exec_tools = [_OPT_MAIN],
        tags = tags,
        **kwargs
    )

    codegen(
        name,
        srcs = [name + "_opt_ir"],
        codegen_params = codegen_params,
        entry = entry,
        tags = tags,
        **kwargs
    )

    # Add a build test to ensure changes to BUILD and bzl files do not break
    # targets built with released toolchains.
    build_test(
        name = name + "_build_test",
        targets = [":" + name],
    )
