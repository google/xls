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
import dataclasses
import enum
import itertools
import subprocess
from typing import Optional, TypeVar

from absl import app
from absl import flags
import jinja2

from xls.common import runfiles
from xls.ir import xls_ir_interface_pb2 as ir_interface_pb2
from xls.ir import xls_type_pb2 as type_pb2
from xls.jit import aot_entrypoint_pb2


_FUNCTION_TYPE = flags.DEFINE_string(
    "function_type",
    default=None,
    required=False,
    help=(
        "If set require a specific type of function. Options are [FUNCTION,"
        " PROC]."
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
        "Function/proc to wrap. If unspecified the top function/proc will be"
        " used. NB If this is a proc it *must* be the top proc unless new-style"
        " procs are being used."
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


@dataclasses.dataclass(frozen=True)
class XlsNamedValue:
  """A Named & typed value for the wrapped function/proc."""

  name: str
  packed_type: str
  unpacked_type: str
  specialized_type: Optional[str]

  @property
  def value_arg(self):
    return f"xls::Value {self.name}"

  @property
  def unpacked_arg(self):
    return f"{self.unpacked_type} {self.name}"

  @property
  def packed_arg(self):
    return f"{self.packed_type} {self.name}"

  @property
  def specialized_arg(self):
    return f"{self.specialized_type} {self.name}"


class JitType(enum.Enum):
  FUNCTION = 1
  PROC = 2


@dataclasses.dataclass(frozen=True)
class XlsChannel:
  xls_name: str
  camel_name: str
  packed_type: str
  unpacked_type: str
  specialized_type: Optional[str]


@dataclasses.dataclass(frozen=True)
class WrappedIr:
  """A wrapped function."""

  jit_type: JitType
  ir_text: str
  function_name: str
  class_name: str
  header_guard: str
  header_filename: str
  namespace: str
  aot_entrypoint: Optional[aot_entrypoint_pb2.AotPackageEntrypointsProto]
  # Function params and result.
  params: Optional[Sequence[XlsNamedValue]] = None
  result: Optional[XlsNamedValue] = None
  # proc state and channels
  incoming_channels: Optional[Sequence[XlsChannel]] = None
  outgoing_channels: Optional[Sequence[XlsChannel]] = None
  state: Optional[Sequence[XlsNamedValue]] = None

  @property
  def can_be_specialized(self) -> bool:
    return (
        all(p.specialized_type is not None for p in self.params)
        and self.result is not None
        and self.result.specialized_type is not None
    )

  @property
  def params_and_result(self):
    return list(self.params) + [self.result]


def to_packed(t: type_pb2.TypeProto) -> str:
  the_type = t.type_enum
  if the_type == type_pb2.TypeProto.BITS:
    return f"xls::PackedBitsView<{t.bit_count}>"
  elif the_type == type_pb2.TypeProto.TUPLE:
    inner = ", ".join(to_packed(e) for e in t.tuple_elements)
    return f"xls::PackedTupleView<{inner}>"
  elif the_type == type_pb2.TypeProto.ARRAY:
    return f"xls::PackedArrayView<{to_packed(t.array_element)}, {t.array_size}>"
  elif the_type == type_pb2.TypeProto.TOKEN:
    return "xls::PackedBitsView<0>"
  raise app.UsageError(
      "Incompatible with argument of type:"
      f" {type_pb2.TypeProto.TypeEnum.Name(t.type_enum)}"
  )


def to_unpacked(t: type_pb2.TypeProto, mutable: bool = False) -> str:
  """Get the unpacked c++ view type.

  Args:
    t: The xls type
    mutable: Whether the type is mutable.

  Returns:
    the C++ unpacked view type
  """
  mutable_str = "Mutable" if mutable else ""
  the_type = t.type_enum
  if the_type == type_pb2.TypeProto.BITS:
    return f"xls::{mutable_str}BitsView<{t.bit_count}>"
  elif the_type == type_pb2.TypeProto.TUPLE:
    inner = ", ".join(to_unpacked(e, mutable) for e in t.tuple_elements)
    return f"xls::{mutable_str}TupleView<{inner}>"
  elif the_type == type_pb2.TypeProto.ARRAY:
    return (
        f"xls::{mutable_str}ArrayView<{to_unpacked(t.array_element, mutable)},"
        f" {t.array_size}>"
    )
  elif the_type == type_pb2.TypeProto.TOKEN:
    return "xls::BitsView<0>"
  raise app.UsageError(
      "Incompatible with argument of type:"
      f" {type_pb2.TypeProto.TypeEnum.Name(t.type_enum)}"
  )


def is_floating_point(
    t: type_pb2.TypeProto, exponent_bits: int, mantissa_bits: int
) -> bool:
  return (
      t.type_enum == type_pb2.TypeProto.TUPLE
      and len(t.tuple_elements) == 3
      and t.tuple_elements[0].type_enum == type_pb2.TypeProto.BITS
      and t.tuple_elements[0].bit_count == 1
      and t.tuple_elements[1].type_enum == type_pb2.TypeProto.BITS
      and t.tuple_elements[1].bit_count == exponent_bits
      and t.tuple_elements[2].type_enum == type_pb2.TypeProto.BITS
      and t.tuple_elements[2].bit_count == mantissa_bits
  )


def is_double_tuple(t: type_pb2.TypeProto) -> bool:
  return is_floating_point(t, 11, 52)


def is_float_tuple(t: type_pb2.TypeProto) -> bool:
  return is_floating_point(t, 8, 23)


def to_specialized(t: type_pb2.TypeProto) -> Optional[str]:
  """Get the specialized c++ type.

  Args:
    t: The xls type

  Returns:
    the C++ type
  """
  the_type = t.type_enum
  if the_type == type_pb2.TypeProto.BITS:
    if t.bit_count <= 8:
      return "uint8_t"
    elif t.bit_count <= 16:
      return "uint16_t"
    elif t.bit_count <= 32:
      return "uint32_t"
    elif t.bit_count <= 64:
      return "uint64_t"
  elif is_double_tuple(t):
    return "double"
  elif is_float_tuple(t):
    return "float"
  elif the_type == type_pb2.TypeProto.ARRAY:
    is_fp = is_float_tuple(t.array_element) or is_double_tuple(t.array_element)
    is_int = (
        t.array_element.type_enum == type_pb2.TypeProto.BITS
        and t.array_element.bit_count in (8, 16, 32, 64)
    )
    # Need to use every bit to match alignment.
    if is_fp or is_int:
      return f"std::array<{to_specialized(t.array_element)}, {t.array_size}>"
    return None
  return None


def to_chan(
    c: ir_interface_pb2.PackageInterfaceProto.Channel, package_name: str
) -> XlsChannel:
  return XlsChannel(
      xls_name=c.name,
      camel_name=camelize(c.name.removeprefix(f"{package_name}__")),
      packed_type=to_packed(c.type),
      unpacked_type=to_unpacked(c.type),
      specialized_type=to_specialized(c.type),
  )


def to_param(
    p: ir_interface_pb2.PackageInterfaceProto.NamedValue,
) -> XlsNamedValue:
  return XlsNamedValue(
      name=p.name,
      packed_type=to_packed(p.type),
      unpacked_type=to_unpacked(p.type),
      specialized_type=to_specialized(p.type),
  )


def interpret_function_interface(
    ir: str,
    func_ir: ir_interface_pb2.PackageInterfaceProto.Function,
    class_name: str,
    header_guard: str,
    header_filename: str,
    aot_info: aot_entrypoint_pb2.AotPackageEntrypointsProto,
) -> WrappedIr:
  """Fill in a WrappedIr for a function from the interface.

  Args:
    ir: package IR
    func_ir: the particular function we want to wrap
    class_name: The class name
    header_guard: The header-guard string
    header_filename: The header file name.
    aot_info: The aot info for the function.

  Returns:
    A wrapped ir for the function.

  Raises:
    UsageError: If the aot info is for a different function.
  """
  if func_ir.base.name != aot_info.entrypoint[0].xls_function_identifier:
    raise app.UsageError("Aot info is for a different function.")
  params = [to_param(p) for p in func_ir.parameters]
  result = XlsNamedValue(
      name="result",
      packed_type=to_packed(func_ir.result_type),
      unpacked_type=to_unpacked(func_ir.result_type, mutable=True),
      specialized_type=to_specialized(func_ir.result_type),
  )
  namespace = _WRAPPER_NAMESPACE.value
  return WrappedIr(
      jit_type=JitType.FUNCTION,
      ir_text=ir,
      function_name=func_ir.base.name,
      class_name=class_name,
      header_guard=header_guard,
      header_filename=header_filename,
      namespace=namespace,
      params=params,
      result=result,
      aot_entrypoint=aot_info,
  )


def interpret_proc_interface(
    ir: str,
    package: ir_interface_pb2.PackageInterfaceProto,
    proc_ir: ir_interface_pb2.PackageInterfaceProto.Proc,
    class_name: str,
    header_guard: str,
    header_filename: str,
    aot_info: aot_entrypoint_pb2.AotPackageEntrypointsProto,
) -> WrappedIr:
  """Fill in a WrappedIr for a proc from the interface.

  Args:
    ir: package IR
    package: the package interface proto
    proc_ir: the particular proc we want to wrap
    class_name: The class name
    header_guard: The header-guard string
    header_filename: The header file name.
    aot_info: The aot info for the function.

  Returns:
    A wrapped ir for the proc.

  Raises:
    UsageError: If the aot info does not contain the top proc.
  """
  top = proc_ir.base.name
  checks = (v.xls_function_identifier != top for v in aot_info.entrypoint)
  if all(checks):
    raise app.UsageError(
        f"AOT info does not contain an entry for top proc {proc_ir.base.name}"
    )
  state = [to_param(p) for p in proc_ir.state]
  input_channels = [to_chan(p, package.name) for p in package.channels]
  output_channels = [to_chan(p, package.name) for p in package.channels]
  return WrappedIr(
      jit_type=JitType.PROC,
      ir_text=ir,
      function_name=proc_ir.base.name,
      class_name=class_name,
      header_guard=header_guard,
      header_filename=header_filename,
      namespace=_WRAPPER_NAMESPACE.value,
      incoming_channels=input_channels,
      outgoing_channels=output_channels,
      state=state,
      aot_entrypoint=aot_info,
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
) -> WrappedIr:
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
      return interpret_function_interface(
          ir,
          func_ir,
          class_name,
          header_guard,
          header_filename,
          aot_info,
      )
  # Try to find a proc
  if _FUNCTION_TYPE.value in (None, "PROC"):
    proc_ir = find_named_entry(interface.procs, function_name, interface.name)
    if proc_ir is not None:
      return interpret_proc_interface(
          ir,
          interface,
          proc_ir,
          class_name,
          header_guard,
          header_filename,
          aot_info,
      )
  raise app.UsageError(
      f"No function/proc called {function_name} in {interface.name} found."
      " options are functions:"
      f" [{', '.join(f.base.name for f in interface.functions)}],  procs:"
      f" [{', '.join(f.base.name for f in interface.procs)}]"
  )


def top_name(ir: ir_interface_pb2.PackageInterfaceProto) -> str:
  for f in itertools.chain(ir.functions, ir.procs, ir.blocks):
    if f.base.top:
      return f.base.name

  raise app.UsageError("--function required if top is not set.")


def camelize(name: str) -> str:
  return name.title().replace("_", "")


_CC_TEMPLATES = {
    JitType.FUNCTION: runfiles.get_contents_as_text(
        "xls/jit/jit_function_wrapper_cc.tmpl"
    ),
    JitType.PROC: runfiles.get_contents_as_text(
        "xls/jit/jit_proc_wrapper_cc.tmpl"
    ),
}

_H_TEMPLATES = {
    JitType.FUNCTION: runfiles.get_contents_as_text(
        "xls/jit/jit_function_wrapper_h.tmpl"
    ),
    JitType.PROC: runfiles.get_contents_as_text(
        "xls/jit/jit_proc_wrapper_h.tmpl"
    ),
}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Incorrect arguments")
  if _FUNCTION_TYPE.value not in (None, "FUNCTION", "PROC"):
    raise app.UsageError(
        "Unknown --function_type. Requires none or FUNCTION or PROC"
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
      _CLASS_NAME.value if _CLASS_NAME.value else camelize(function_name)
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
