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

"""Dump a c-main function which can be used to drive a compiled XLS function.

This is meant for use with llvm tools such as bugpoint, lli, & llvm-link to
diagnose and investigate LLVM bugs.
"""

from collections.abc import Sequence
import dataclasses
import functools
import subprocess
import tempfile

from absl import app
from absl import flags
import jinja2

from xls.common import runfiles
from xls.jit import aot_entrypoint_pb2
from xls.jit import type_layout_pb2

_TYPE_LAYOUT_BIN = "xls/jit/type_layout_main"

_INPUT = flags.DEFINE_multi_string(
    "input",
    default=[],
    help=(
        'Each input to have the main function For example:  "(bits[7]:0,'
        ' bits[20]:4)"'
    ),
)
_RESULT = flags.DEFINE_string(
    "result",
    default=None,
    required=False,
    help=(
        "Value to expect the result to be. Formatted the same as 'input' though"
        " only a single value."
    ),
)
_OUTPUT = flags.DEFINE_string(
    "output",
    default=None,
    required=True,
    help="File to write the main cc file to.",
)
_PACKAGE = flags.DEFINE_string(
    "package",
    default=None,
    required=True,
    help="AotPackageEntrypointsProto that is being called",
)
_TOP = flags.DEFINE_string(
    "top",
    default=None,
    required=False,
    help=(
        "The entrypoint to actually call, defaults to only one present if not"
        " given."
    ),
)
_WRITE_RESULT = flags.DEFINE_bool(
    "write_result",
    default=False,
    required=False,
    help=(
        "Add a write of the resulting value (requires that there be a 'write'"
        " symbol for linking so incompatible with non-jit llvm interp)."
    ),
)
_MASK = flags.DEFINE_string(
    "mask",
    default=None,
    required=False,
    help="Mask value to use when checking results",
)


def _to_type_mask(proto: type_layout_pb2.TypeLayoutProto) -> bytes:
  """Get the byte representation of the mask for the given type-layout.

  Args:
    proto: The typelayout the value must conform to

  Returns:
    The serialized version of a mask denoting which bits of an output are
    meaningful.
  """
  with tempfile.NamedTemporaryFile(mode="wb") as proto_file:
    proto_file.write(proto.SerializeToString())
    proto_file.flush()
    res = subprocess.run(
        [
            runfiles.get_path(_TYPE_LAYOUT_BIN),
            "-layout_proto",
            f"{proto_file.name}",
            "-mask",
        ],
        check=True,
        encoding=None,
        stdout=subprocess.PIPE,
    )
    return res.stdout


def _to_type_layout(proto: type_layout_pb2.TypeLayoutProto, v: str) -> bytes:
  """Get the byte representation of the value in the given type layout.

  Args:
    proto: The typelayout the value must conform to
    v: The value to serialize

  Returns:
    The serialized version of 'v' in the layout
  """
  with tempfile.NamedTemporaryFile(mode="wb") as proto_file:
    proto_file.write(proto.SerializeToString())
    proto_file.flush()
    res = subprocess.run(
        [
            runfiles.get_path(_TYPE_LAYOUT_BIN),
            "-layout_proto",
            f"{proto_file.name}",
            "-encode",
            f"{v}",
        ],
        check=True,
        encoding=None,
        stdout=subprocess.PIPE,
    )
    return res.stdout


@dataclasses.dataclass(frozen=True)
class _AotArg:
  """Information about an argument to an AOT entrypoint.

  Attributes:
    layout: The layout of the argument.
    value_str: The value of the argument.
    alignment: The alignment of the argument.
    name: The name of the argument.
    expected_size: The expected size of the argument.
    value: The value of the argument in bytes.
  """

  layout: type_layout_pb2.TypeLayoutProto
  value_str: str
  alignment: int
  name: str
  expected_size: int

  @functools.cached_property
  def value(self) -> bytes:
    res = _to_type_layout(self.layout, self.value_str)
    assert len(res) == self.expected_size
    return res


@dataclasses.dataclass(frozen=True)
class _AotInfo:
  """Information about the AOT entrypoint to be called.

  Attributes:
    entrypoint: The entrypoint to be called.
    result_str: The expected result of the entrypoint.
    result_mask_str: The mask to use when checking the result.
    args_str: The arguments to the entrypoint.
    write_result: Whether to write the result to the output buffer.
    extern_fn: The name of the extern function to call.
    expected_result: The expected result of the entrypoint.
    temp_alignment: The alignment of the temp buffer.
    temp_size: The size of the temp buffer.
    result_alignment: The alignment of the result buffer.
    expected_result_mask: The mask to use when checking the result.
    args: The arguments to the entrypoint.
  """

  entrypoint: aot_entrypoint_pb2.AotEntrypointProto
  result_str: str
  result_mask_str: str
  args_str: Sequence[str]
  write_result: bool

  @property
  def extern_fn(self) -> str:
    return self.entrypoint.function_symbol

  @functools.cached_property
  def expected_result(self) -> bytes:
    res = _to_type_layout(
        self.entrypoint.outputs_layout.layouts[0], self.result_str
    )
    assert len(res) == self.entrypoint.output_buffer_sizes[0]
    return res

  @property
  def temp_alignment(self) -> int:
    return self.entrypoint.temp_buffer_alignment

  @property
  def temp_size(self) -> int:
    return self.entrypoint.temp_buffer_size

  @property
  def result_alignment(self) -> int:
    return self.entrypoint.output_buffer_alignments[0]

  @functools.cached_property
  def expected_result_mask(self) -> bytes:
    if self.result_mask_str is None:
      return _to_type_mask(self.entrypoint.outputs_layout.layouts[0])
    else:
      return _to_type_layout(
          self.entrypoint.outputs_layout.layouts[0], self.result_mask_str
      )

  @functools.cached_property
  def args(self) -> Sequence[_AotArg]:
    return [
        _AotArg(
            layout=layout,
            value_str=value,
            alignment=align,
            name=name,
            expected_size=size,
        )
        for layout, value, align, name, size in zip(
            self.entrypoint.inputs_layout.layouts,
            self.args_str,
            self.entrypoint.input_buffer_alignments,
            self.entrypoint.inputs_names,
            self.entrypoint.input_buffer_sizes,
        )
    ]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with open(_PACKAGE.value, "rb") as ep:
    all_entrypoints = aot_entrypoint_pb2.AotPackageEntrypointsProto.FromString(
        ep.read()
    )
  if not _TOP.value:
    if len(all_entrypoints.entrypoint) != 1:
      raise app.UsageError(
          "Multiple entrypoints possible, --entrypoint must be specified"
      )
    entrypoint = all_entrypoints.entrypoint[0]
  else:

    def is_target(a: aot_entrypoint_pb2.AotEntrypointProto) -> bool:
      return (
          _TOP.value == a.xls_function_identifier
          or f"__{a.xls_package_name}__{_TOP.value}"
          == a.xls_function_identifier
      )

    poss = [a for a in all_entrypoints.entrypoint if is_target(a)]
    if len(poss) != 1:
      raise app.UsageError(f"Multiple possible entrypoints: {poss}")
    entrypoint = poss[0]
  if entrypoint.type != aot_entrypoint_pb2.AotEntrypointProto.FUNCTION:
    raise app.UsageError("Non-function entrypoints not supported.")
  aot = _AotInfo(
      entrypoint=entrypoint,
      result_str=_RESULT.value,
      result_mask_str=_MASK.value,
      args_str=_INPUT.value,
      write_result=_WRITE_RESULT.value,
  )
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  bindings = {"aot": aot, "len": len}
  with open(_OUTPUT.value, "wt") as cc_file:
    cc_file.write("// Generated File. Do not edit.\n")
    cc_file.write(
        env.from_string(
            runfiles.get_contents_as_text("xls/jit/aot_main_cc.tmpl")
        ).render(bindings)
    )


if __name__ == "__main__":
  app.run(main)
