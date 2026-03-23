# Copyright 2026 The XLS Authors
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

"""Library for creating a jit-wrapper impl based on the IR."""

from collections.abc import Sequence
import dataclasses
import enum
from typing import Optional

from absl import app
import jinja2

from xls.ir import xls_ir_interface_pb2 as ir_interface_pb2
from xls.ir import xls_type_pb2 as type_pb2
from xls.jit import aot_entrypoint_pb2


@dataclasses.dataclass(frozen=True)
class XlsNamedValue:
  """A Named & typed value for the wrapped function/proc."""

  name: str
  packed_type: str
  unpacked_type: str
  specialized_type: Optional[str]
  type_proto: type_pb2.TypeProto

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
  BLOCK = 3
  FUZZTEST = 4


@dataclasses.dataclass(frozen=True)
class XlsChannel:
  xls_name: str
  proc_name: str
  camel_name: str
  packed_type: str
  unpacked_type: str
  specialized_type: Optional[str]


@dataclasses.dataclass(frozen=True)
class XlsPort:
  xls_name: str
  camel_name: str
  snake_name: str
  bit_count: Optional[int]
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
  # block ports
  input_ports: Optional[Sequence[XlsPort]] = None
  output_ports: Optional[Sequence[XlsPort]] = None

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


@dataclasses.dataclass(frozen=True)
class PropertyFunctionParam:
  """A fuzztest property function parameter. This will be fuzzed by FuzzTest."""

  name: str
  # The index of the parameter in the function signature.
  index: int


@dataclasses.dataclass(frozen=True)
class PropertyFunction:
  """A fuzztest property function."""

  fuzztest_name: str
  property_function_name: str
  lib_header_path: str
  lib_class_name: str
  namespace: str
  # Function params and result.
  params: Sequence[PropertyFunctionParam]
  return_type: bool


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


def to_c_type(t: type_pb2.TypeProto) -> str:
  c_type = to_specialized(t, int_only=True)
  if c_type is not None:
    return c_type
  the_type = t.type_enum
  if the_type == type_pb2.TypeProto.TUPLE:
    inner = ", ".join(to_c_type(e) for e in t.tuple_elements)
    return f"std::tuple<{inner}>"
  elif the_type == type_pb2.TypeProto.ARRAY:
    return f"std::array<{to_c_type(t.array_element)}, {t.array_size}>"
  raise app.UsageError(f"Cannot convert {t} to c_type")


def to_specialized(
    t: type_pb2.TypeProto, *, int_only: bool = False
) -> Optional[str]:
  """Get the specialized c++ type.

  Args:
    t: The xls type
    int_only: If true only allow 'integer/bits' typed things to be specialized

  Returns:
    the C++ type
  """
  the_type = t.type_enum
  if int_only and the_type != type_pb2.TypeProto.BITS:
    return None
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
      proc_name=c.proc_name,
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
      type_proto=p.type,
  )


def to_port(
    p: ir_interface_pb2.PackageInterfaceProto.NamedValue, package_name: str
) -> XlsPort:
  return XlsPort(
      xls_name=p.name,
      camel_name=camelize(p.name.removeprefix(f"{package_name}__")),
      snake_name=p.name.removeprefix(f"{package_name}__"),
      bit_count=p.type.bit_count
      if p.type.type_enum == type_pb2.TypeProto.BITS
      else None,
      specialized_type=to_specialized(p.type, int_only=True),
  )


def interpret_function_interface(
    ir: str,
    func_ir: ir_interface_pb2.PackageInterfaceProto.Function,
    class_name: str,
    header_guard: str,
    header_filename: str,
    aot_info: aot_entrypoint_pb2.AotPackageEntrypointsProto,
    wrapper_namespace: str,
) -> WrappedIr:
  """Fill in a WrappedIr for a function from the interface.

  Args:
    ir: package IR
    func_ir: the particular function we want to wrap
    class_name: The class name
    header_guard: The header-guard string
    header_filename: The header file name.
    aot_info: The aot info for the function.
    wrapper_namespace: The namespace to wrap the code in.

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
      type_proto=func_ir.result_type,
  )
  namespace = wrapper_namespace
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
    wrapper_namespace: str,
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
    wrapper_namespace: The namespace to wrap the code in.

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
  state = [to_param(p.name) for p in proc_ir.state_values]
  input_channels = [to_chan(p, package.name) for p in package.channels]
  output_channels = [to_chan(p, package.name) for p in package.channels]
  return WrappedIr(
      jit_type=JitType.PROC,
      ir_text=ir,
      function_name=proc_ir.base.name,
      class_name=class_name,
      header_guard=header_guard,
      header_filename=header_filename,
      namespace=wrapper_namespace,
      incoming_channels=input_channels,
      outgoing_channels=output_channels,
      state=state,
      aot_entrypoint=aot_info,
  )


def interpret_block_interface(
    ir: str,
    package: ir_interface_pb2.PackageInterfaceProto,
    block_ir: ir_interface_pb2.PackageInterfaceProto.Block,
    class_name: str,
    header_guard: str,
    header_filename: str,
    aot_info: aot_entrypoint_pb2.AotPackageEntrypointsProto,
    wrapper_namespace: str,
) -> WrappedIr:
  """Fill in a WrappedIr for a block from the interface.

  Args:
    ir: package IR
    package: the package interface proto
    block_ir: the particular block we want to wrap
    class_name: The class name
    header_guard: The header-guard string
    header_filename: The header file name.
    aot_info: The aot info for the function.
    wrapper_namespace: The namespace to wrap the code in.

  Returns:
    A wrapped ir for the block.
  """
  input_ports = [to_port(p, package.name) for p in block_ir.input_ports]
  output_ports = [to_port(p, package.name) for p in block_ir.output_ports]
  namespace = wrapper_namespace
  return WrappedIr(
      jit_type=JitType.BLOCK,
      ir_text=ir,
      function_name=block_ir.base.name,
      class_name=class_name,
      header_guard=header_guard,
      header_filename=header_filename,
      namespace=namespace,
      input_ports=input_ports,
      output_ports=output_ports,
      aot_entrypoint=aot_info,
  )


def camelize(name: str) -> str:
  return name.title().replace("_", "")


def wrapped_to_fuzztest(
    wrapped: WrappedIr, lib_class_name: str, lib_header_path: str
) -> PropertyFunction:
  """Converts a WrappedIr object to a dictionary for fuzztest template."""
  params = []
  if wrapped.params:
    for idx, p in enumerate(wrapped.params):
      params.append(PropertyFunctionParam(name=p.name, index=idx))
  return PropertyFunction(
      fuzztest_name=wrapped.function_name + "_fuzztest",
      property_function_name=wrapped.function_name,
      lib_class_name=lib_class_name,
      lib_header_path=lib_header_path,
      # Everything shares the namespace
      namespace=wrapped.namespace,
      params=params,
      return_type=wrapped.result is not None,
  )


def render_fuzztest(
    wrapped: WrappedIr,
    env: jinja2.Environment,
    cc_template_content: str,
    lib_class_name: str,
    lib_header_path: str,
) -> str:
  """Renders the fuzztest C++ code."""
  env.filters["property_param"] = lambda p: "xls::Value " + p.name
  cc_template = env.from_string(cc_template_content)
  fuzztest = wrapped_to_fuzztest(wrapped, lib_class_name, lib_header_path)
  bindings = {"fuzztest": fuzztest, "len": len}
  return cc_template.render(bindings)
