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

"""Compatibility implementation of the old aot_compiler API.

This generates a single entrypoint for an XLS function.
"""

from collections.abc import Sequence
import dataclasses

from absl import app
from absl import flags
import jinja2

from google.protobuf import text_format
from xls.common import runfiles
from xls.jit import aot_entrypoint_pb2
from xls.jit import type_layout_pb2


_READ_TEXTPROTO = flags.DEFINE_bool(
    "read_textproto",
    default=False,
    required=False,
    help="Read the input AotEntrypointProto as a text-proto",
)
_NAMESPACES = flags.DEFINE_string(
    "namespaces",
    default="",
    help=(
        "Comma-separated list of namespaces into which to place the generated"
        " code. Earlier-specified namespaces enclose later-specified."
    ),
    required=False,
)
_OUTPUT_HEADER = flags.DEFINE_string(
    "output_header",
    default=None,
    required=True,
    help="Path at which to write the output header file.",
)
_OUTPUT_SOURCE = flags.DEFINE_string(
    "output_source",
    default=None,
    required=True,
    help="Path at which to write the output object-wrapping source file.",
)
_HEADER_INCLUDE_PATH = flags.DEFINE_string(
    "header_include_path",
    default=None,
    required=True,
    help=(
        "The path in the source tree at which the header should be #included."
        " This is copied verbatim into an #include directive in the generated"
        " source file (the .cc file specified with --output_source). This flag"
        " is required."
    ),
)


@dataclasses.dataclass(frozen=True)
class AotParameter:
  """AOT data for a function parameter.

  Attributes:
    name: The name of the parameter.
    size: The size of the parameter.
    preferred_alignment: The preferred alignment of the parameter.
  """

  name: str
  size: int
  preferred_alignment: int

  @property
  def param(self) -> str:
    """The parameter as a C++ declaration."""
    return f"::xls::Value {self.name}"


@dataclasses.dataclass(frozen=True)
class BasicFunctionAot:
  """AOT data for a basic function.

  Attributes:
    namespace: The namespace in which to place the generated code.
    header_filename: The filename of the header file to be #included.
    extern_fn: The name of the function to be called.
    wrapper_function_name: The name of the wrapper function to be generated.
    arguments: The arguments to the wrapper function.
    result_buffer_align: The alignment of the result buffer.
    result_buffer_size: The size of the result buffer.
    temp_buffer_align: The alignment of the temp buffer.
    temp_buffer_size: The size of the temp buffer.
    arg_layout: The TypeLayout of the arguments.
    result_layout: The TypeLayout of the result.
    extern_sanitizer: Whether msan is linked in.
  """

  namespace: str
  header_filename: str
  extern_fn: str
  wrapper_function_name: str
  arguments: Sequence[AotParameter]
  result_buffer_align: int
  result_buffer_size: int
  temp_buffer_align: int
  temp_buffer_size: int
  arg_layout: type_layout_pb2.TypeLayoutsProto
  result_layout: type_layout_pb2.TypeLayoutsProto
  extern_sanitizer: bool


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    raise app.UsageError(f"Usage: {argv[0]} [flags] AotEntrypointProto")
  if _READ_TEXTPROTO.value:
    entrypoint = aot_entrypoint_pb2.AotEntrypointProto()
    with open(argv[1], "rt") as proto:
      text_format.Parse(proto.read(), entrypoint)
  else:
    with open(argv[1], "rb") as proto:
      entrypoint = aot_entrypoint_pb2.AotEntrypointProto.FromString(
          proto.read()
      )
  params = []
  for name, size, align in zip(
      entrypoint.inputs_names,
      entrypoint.input_buffer_sizes,
      entrypoint.input_buffer_alignments,
  ):
    params.append(AotParameter(name, size, align))
  aot = BasicFunctionAot(
      namespace="::".join(_NAMESPACES.value.split(",")),
      header_filename=_HEADER_INCLUDE_PATH.value,
      extern_fn=entrypoint.function_symbol,
      wrapper_function_name=entrypoint.xls_function_identifier.removeprefix(
          f"__{entrypoint.xls_package_name}__"
      ),
      arguments=params,
      result_buffer_align=entrypoint.output_buffer_alignments[0],
      result_buffer_size=entrypoint.output_buffer_sizes[0],
      temp_buffer_align=entrypoint.temp_buffer_alignment,
      temp_buffer_size=entrypoint.temp_buffer_size,
      arg_layout=entrypoint.inputs_layout,
      result_layout=entrypoint.outputs_layout,
      extern_sanitizer=entrypoint.has_msan,
  )
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  env.filters["append_each"] = lambda vs, suffix: [v + suffix for v in vs]
  bindings = {
      "aot": aot,
      "len": len,
      "MessageToString": text_format.MessageToString,
  }
  with open(_OUTPUT_HEADER.value, "wt") as h_file:
    h_template = env.from_string(
        runfiles.get_contents_as_text("xls/jit/aot_basic_function_h.tmpl")
    )
    h_file.write("// Generated File. Do not edit.\n")
    h_file.write(h_template.render(bindings))
    h_file.write("\n")
  with open(_OUTPUT_SOURCE.value, "wt") as cc_file:
    cc_template = env.from_string(
        runfiles.get_contents_as_text("xls/jit/aot_basic_function_cc.tmpl")
    )
    cc_file.write("// Generated File. Do not edit.\n")
    cc_file.write(cc_template.render(bindings))
    cc_file.write("\n")


if __name__ == "__main__":
  app.run(main)
