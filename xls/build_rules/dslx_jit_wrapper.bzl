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

"""See dslx_jit_wrapper()."""

load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")

# TODO(rspringer): 2020-08-05 The dslx_test macro is a bit packed, which makes
# declaring dslx_jit_wrapper rules a bit torturous - one needs to depend on an
# IR target instead of a testonly (!) dslx_test target. When that's done, we
# can stop depending on an explicit _opt_ir target in invocations.
def dslx_jit_wrapper(
        name,
        dslx_name = None,
        entry_function = None,
        deps = [],
        **kwargs):
    """Generates sources and rules for JIT invocation wrappers.

    Args:
      name: The name of the dslx target being wrapped.
      dslx_name: Name of the generated class. If unspecified, the
        entry function name will be used (see 'entry function' below).
      entry_function: The name of the function being wrapped. If
        unspecified, the standard entry function lookup will be performed (
        searches for functions w/the package name or "main" or some variations
        thereof).
      deps: Dependencies of this wrapper - likely only the source IR.
      **kwargs: Extra arguments to pass to genrule.
    """
    entry_arg = ("--function=" + entry_function) if entry_function else ""
    genrule_wrapper(
        name = "gen_" + name,
        srcs = deps,
        outs = [
            name + ".h",
            name + ".cc",
        ],
        cmd = "$(location //xls/jit:jit_wrapper_generator_main) -ir_path $(SRCS) %s -class_name %s -output_name %s -output_dir $(@D) -genfiles_dir $(GENDIR)" % (entry_arg, dslx_name, name),
        exec_tools = [
            "//xls/jit:jit_wrapper_generator_main",
        ],
        **kwargs
    )

    native.cc_library(
        name = name,
        srcs = [name + ".cc"],
        hdrs = [name + ".h"],
        deps = [
            "@com_google_absl//absl/status",
            "//xls/common/status:status_macros",
            "@com_google_absl//absl/status:statusor",
            "//xls/public:function_builder",
            "//xls/public:value",
            "//xls/ir:ir_parser",
            "//xls/jit:ir_jit",
        ],
    )
