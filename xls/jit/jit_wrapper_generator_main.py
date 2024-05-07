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
import subprocess
from typing import Optional

from absl import app
from absl import flags
import jinja2

from xls.common import runfiles
from xls.ir import xls_ir_interface_pb2 as ir_interface_pb2
from xls.ir import xls_type_pb2 as type_pb2


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
    help="Function to wrap. If unspecified the top function will be used.",
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


@dataclasses.dataclass(frozen=True)
class XlsParam:
  """A parameter for the wrapped function."""

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


@dataclasses.dataclass(frozen=True)
class WrappedIr:
  """A wrapped function."""

  ir_text: str
  function_name: str
  class_name: str
  header_guard: str
  header_filename: str
  namespace: str
  params: Sequence[XlsParam]
  result: XlsParam

  @property
  def can_be_specialized(self) -> bool:
    return (
        all(p.specialized_type is not None for p in self.params)
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
  return None


def to_param(p: ir_interface_pb2.PackageInterfaceProto.NamedValue) -> XlsParam:
  return XlsParam(
      name=p.name,
      packed_type=to_packed(p.type),
      unpacked_type=to_unpacked(p.type),
      specialized_type=to_specialized(p.type),
  )


def interpret_interface(
    ir: str,
    interface: ir_interface_pb2.PackageInterfaceProto,
    output_name: str,
    class_name: str,
    function_name: str,
) -> WrappedIr:
  """Create a wrapped-ir representation of the IR to be rendered to source.

  Args:
    ir: The XLS ir being wrapped
    interface: The interface proto describing the contents of the IR
    output_name: what the file basename we are writing to is.
    class_name: what the class we are creating is called.
    function_name: what the IR function we are actually calling is.

  Returns:
    A WrappedIr ready for rendering.

  Raises:
    UsageError: If the IR/interface does not contain an appropriate function.
  """
  func_ir = None
  for f in interface.functions:
    if (
        f.base.name == function_name
        or f.base.name.removeprefix(f"__{interface.name}__") == function_name
    ):
      func_ir = f
      break
  if func_ir is None:
    raise app.UsageError(
        f"No function called {function_name} in {interface.name} found. options"
        f" are: [{', '.join(f.base.name for f in interface.functions)}]",
    )
  params = [to_param(p) for p in func_ir.parameters]
  result = XlsParam(
      name="result",
      packed_type=to_packed(func_ir.result_type),
      unpacked_type=to_unpacked(func_ir.result_type, mutable=True),
      specialized_type=to_specialized(func_ir.result_type),
  )
  header_guard = (
      f"{_OUTPUT_DIR.value}/{output_name}_H_"[len(_GENFILES_DIR.value) :]
      .replace("/", "_")  # Get rid if bazel-gen/...
      .capitalize()
  )
  namespace = _WRAPPER_NAMESPACE.value
  return WrappedIr(
      ir_text=ir,
      function_name=func_ir.base.name,
      class_name=class_name,
      header_guard=header_guard,
      header_filename=f"{_OUTPUT_DIR.value}/{output_name}.h",
      namespace=namespace,
      params=params,
      result=result,
  )


def top_function_name(ir: ir_interface_pb2.PackageInterfaceProto) -> str:
  for f in ir.functions:
    if f.base.top:
      return f.base.name
  raise app.UsageError("--function required if top is not set.")


def camelize(name: str) -> str:
  return name.title().replace("_", "")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Incorrect arguments")
  ir_interface = ir_interface_pb2.PackageInterfaceProto.FromString(
      subprocess.check_output([
          runfiles.get_path("xls/tools/extract_interface_main"),
          "--binary_proto",
          _IR_PATH.value,
      ])
  )
  function_name = (
      _FUNCTION.value if _FUNCTION.value else top_function_name(ir_interface)
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
  )

  # Create the JINJA env and add an append filter.
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  env.filters["append_each"] = lambda vs, suffix: [v + suffix for v in vs]
  bindings = {"wrapped": wrapped, "len": len}

  with open(f"{_OUTPUT_DIR.value}/{output_name}.cc", "wt") as cc_file:
    cc_template = env.from_string(
        runfiles.get_contents_as_text("xls/jit/jit_function_wrapper_cc.tmpl")
    )
    cc_file.write("// Generated File. Do not edit.\n")
    cc_file.write(cc_template.render(bindings))
    cc_file.write("\n")
  with open(f"{_OUTPUT_DIR.value}/{output_name}.h", "wt") as h_file:
    h_template = env.from_string(
        runfiles.get_contents_as_text("xls/jit/jit_function_wrapper_h.tmpl")
    )
    h_file.write("// Generated File. Do not edit.\n")
    h_file.write(h_template.render(bindings))
    h_file.write("\n")


if __name__ == "__main__":
  app.run(main)
