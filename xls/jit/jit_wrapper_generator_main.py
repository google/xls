# Copyright 2024 The XLS Authors
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

"""Creates a jit-wrapper impl based on the IR.

Output source code is saved according to output_name
"""

from collections.abc import Sequence
import itertools
import subprocess
from typing import Optional, TypeVar

from absl import app
from absl import flags
import jinja2

from xls.common import runfiles
from xls.ir import xls_ir_interface_pb2 as ir_interface_pb2
from xls.jit import aot_entrypoint_pb2
from xls.jit import jit_wrapper_generator


_FUNCTION_TYPE = flags.DEFINE_string(
    "function_type",
    default=None,
    required=False,
    help=(
        "If set require a specific type of function. Options are [FUNCTION,"
        " PROC, BLOCK]."
    ),
)
_CLASS_NAME = flags.DEFINE_string(
    "class_name",
    default="",
    required=False,
    help=(
        "Name of the generated class. If unspecified, the camelized wrapped"
        " function name will be used."
    ),
)
_FUNCTION = flags.DEFINE_string(
    "function",
    default=None,
    required=False,
    help=(
        "Function/proc/block to wrap. If unspecified the top function/proc will"
        " be used. NB If this is a proc it *must* be the top proc unless"
        " new-style procs are being used."
    ),
)
_IR_PATH = flags.DEFINE_string(
    "ir_path", default=None, required=True, help="Path to the IR to wrap."
)
_OUTPUT_NAME = flags.DEFINE_string(
    "output_name",
    default=None,
    required=False,
    help=(
        "Name of the generated files, foo.h and foo.c. If unspecified, the"
        " wrapped function name will be used."
    ),
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    required=True,
    default=None,
    help=(
        "Directory into which to write the output. Files will be named"
        " <function>.h and <function>.cc"
    ),
)
_GENFILES_DIR = flags.DEFINE_string(
    "genfiles_dir",
    required=True,
    default=None,
    help=(
        "The directory into which generated files are placed. "
        "This prefix will be removed from the header guards."
    ),
)
_WRAPPER_NAMESPACE = flags.DEFINE_string(
    "wrapper_namespace",
    required=False,
    default="xls",
    help="C++ namespace to put the wrapper in.",
)
_AOT_INFO = flags.DEFINE_string(
    "aot_info",
    required=True,
    default=None,
    help=(
        "Proto file describing the interface of the available AOT'd functions"
        " as a AotPackageEntrypointsProto. Must be a binary proto."
    ),
)


_T = TypeVar("_T")


def find_named_entry(
    lst: Sequence[_T], function_name: str, interface_name: str
) -> Optional[_T]:
  for f in lst:
    if (
        f.base.name == function_name
        or f.base.name.removeprefix(f"__{interface_name}__") == function_name
    ):
      return f
  return None


def interpret_interface(
    ir: str,
    interface: ir_interface_pb2.PackageInterfaceProto,
    output_name: str,
    class_name: str,
    function_name: str,
    aot_info: aot_entrypoint_pb2.AotPackageEntrypointsProto,
) -> jit_wrapper_generator.WrappedIr:
  """Create a wrapped-ir representation of the IR to be rendered to source.

  Args:
    ir: The XLS ir being wrapped
    interface: The interface proto describing the contents of the IR
    output_name: what the file basename we are writing to is.
    class_name: what the class we are creating is called.
    function_name: what the IR function we are actually calling is.
    aot_info: The aot info for the function.

  Returns:
    A WrappedIr ready for rendering.

  Raises:
    UsageError: If the IR/interface does not contain an appropriate function.
  """
  # Try to find a function
  header_guard = (
      f"{_OUTPUT_DIR.value}/{output_name}_H_"[len(_GENFILES_DIR.value) :]
      .replace("/", "_")  # Get rid if bazel-gen/...
      .replace("-", "_")
      .upper()
  )
  header_filename = f"{_OUTPUT_DIR.value}/{output_name}.h"
  if _FUNCTION_TYPE.value in (None, "FUNCTION"):
    func_ir = find_named_entry(
        interface.functions,
        function_name,
        interface.name,
    )
    if func_ir is not None:
      return jit_wrapper_generator.interpret_function_interface(
          ir,
          func_ir,
          class_name,
          header_guard,
          header_filename,
          aot_info,
          _WRAPPER_NAMESPACE.value,
      )
  # Try to find a proc
  if _FUNCTION_TYPE.value in (None, "PROC"):
    proc_ir = find_named_entry(interface.procs, function_name, interface.name)
    if proc_ir is not None:
      return jit_wrapper_generator.interpret_proc_interface(
          ir,
          interface,
          proc_ir,
          class_name,
          header_guard,
          header_filename,
          aot_info,
          _WRAPPER_NAMESPACE.value,
      )
  # Try to find a block
  if _FUNCTION_TYPE.value in (None, "BLOCK"):
    block_ir = find_named_entry(interface.blocks, function_name, interface.name)
    if block_ir is not None:
      return jit_wrapper_generator.interpret_block_interface(
          ir,
          interface,
          block_ir,
          class_name,
          header_guard,
          header_filename,
          aot_info,
          _WRAPPER_NAMESPACE.value,
      )
  raise app.UsageError(
      f"No function/proc/block called {function_name} in"
      f" {interface.name} found. options are functions:"
      f" [{', '.join(f.base.name for f in interface.functions)}],  procs:"
      f" [{', '.join(f.base.name for f in interface.procs)}], blocks:"
      f" [{', '.join(f.base.name for f in interface.blocks)}]"
  )


def top_name(ir: ir_interface_pb2.PackageInterfaceProto) -> str:
  for f in itertools.chain(ir.functions, ir.procs, ir.blocks):
    if f.base.top:
      return f.base.name

  raise app.UsageError("--function required if top is not set.")


_CC_TEMPLATES = {
    jit_wrapper_generator.JitType.FUNCTION: runfiles.get_contents_as_text(
        "xls/jit/jit_function_wrapper_cc.tmpl"
    ),
    jit_wrapper_generator.JitType.PROC: runfiles.get_contents_as_text(
        "xls/jit/jit_proc_wrapper_cc.tmpl"
    ),
    jit_wrapper_generator.JitType.BLOCK: runfiles.get_contents_as_text(
        "xls/jit/jit_block_wrapper_cc.tmpl"
    ),
}

_H_TEMPLATES = {
    jit_wrapper_generator.JitType.FUNCTION: runfiles.get_contents_as_text(
        "xls/jit/jit_function_wrapper_h.tmpl"
    ),
    jit_wrapper_generator.JitType.PROC: runfiles.get_contents_as_text(
        "xls/jit/jit_proc_wrapper_h.tmpl"
    ),
    jit_wrapper_generator.JitType.BLOCK: runfiles.get_contents_as_text(
        "xls/jit/jit_block_wrapper_h.tmpl"
    ),
}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Incorrect arguments")
  if _FUNCTION_TYPE.value not in (None, "FUNCTION", "PROC", "BLOCK"):
    raise app.UsageError(
        "Unknown --function_type. Requires none or FUNCTION, BLOCK, or PROC"
    )
  with open(_AOT_INFO.value, "rb") as aot_info_file:
    aot_info = aot_entrypoint_pb2.AotPackageEntrypointsProto.FromString(
        aot_info_file.read()
    )
  ir_interface = ir_interface_pb2.PackageInterfaceProto.FromString(
      subprocess.check_output([
          runfiles.get_path("xls/dev_tools/extract_interface_main"),
          "--binary_proto",
          _IR_PATH.value,
      ])
  )
  function_name = (
      _FUNCTION.value if _FUNCTION.value else top_name(ir_interface)
  ).removeprefix(f"__{ir_interface.name}__")
  class_name = (
      _CLASS_NAME.value
      if _CLASS_NAME.value
      else jit_wrapper_generator.camelize(function_name)
  )
  output_name = _OUTPUT_NAME.value if _OUTPUT_NAME.value else function_name
  wrapped = interpret_interface(
      open(_IR_PATH.value, "rt").read(),
      ir_interface,
      output_name,
      class_name,
      function_name,
      aot_info,
  )

  # Create the JINJA env and add an append filter.
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  env.filters["append_each"] = lambda vs, suffix: [v + suffix for v in vs]
  env.filters["prefix_each"] = lambda vs, prefix: [prefix + v for v in vs]
  env.filters["to_char_ints"] = lambda v: [x for x in v]
  bindings = {"wrapped": wrapped, "len": len, "str": str}

  with open(f"{_OUTPUT_DIR.value}/{output_name}.cc", "wt") as cc_file:
    cc_template = env.from_string(_CC_TEMPLATES[wrapped.jit_type])
    cc_file.write("// Generated File. Do not edit.\n")
    cc_file.write(cc_template.render(bindings))
    cc_file.write("\n")
  with open(f"{_OUTPUT_DIR.value}/{output_name}.h", "wt") as h_file:
    h_template = env.from_string(_H_TEMPLATES[wrapped.jit_type])
    h_file.write("// Generated File. Do not edit.\n")
    h_file.write(h_template.render(bindings))
    h_file.write("\n")


if __name__ == "__main__":
  app.run(main)
