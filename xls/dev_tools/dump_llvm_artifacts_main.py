#
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

"""Generate a set of files which are useful for diagnosing llvm issues on an xls function."""

from collections.abc import Sequence
import itertools
import pathlib
import subprocess

from absl import app
from absl import flags

from xls.common import runfiles
from xls.jit import aot_entrypoint_pb2

_OUT_DIR = flags.DEFINE_string(
    "out_dir",
    default=None,
    required=True,
    help="Directory to put all generated files in.",
)

_IR = flags.DEFINE_string(
    "ir",
    default=None,
    required=True,
    help="IR to generate a wrapper for",
)

_TOP = flags.DEFINE_string(
    "top",
    default=None,
    required=False,
    help="Top function to generate test for",
)

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
_MASK = flags.DEFINE_string(
    "mask", default=None, required=False, help="Mask to apply to result."
)
_WRITE_RESULT = flags.DEFINE_bool(
    "write_result",
    default=False,
    required=False,
    help=(
        "Whether to have the main function write the result buffer to stdout,"
        " requires that a posix 'write' is linked."
    ),
)

_DEFAULT_CLANG = "clang/clang"
_DEFAULT_LLVM_LINK = "llvm/llvm-link"

_CLANG = flags.DEFINE_string(
    "clang",
    default=runfiles.get_path(_DEFAULT_CLANG, repository="llvm-project"),
    required=False,
    help="If present the clang to use to turn the main.cc into an llvm ir",
)
_LLVM_LINK = flags.DEFINE_string(
    "llvm_link",
    default=runfiles.get_path(_DEFAULT_LLVM_LINK, repository="llvm-project"),
    required=False,
    help="If present the llvm-link to use to link the main.cc",
)

_DUMP_MAIN = "xls/jit/aot_main_wrapper_main"
_AOT_COMPILER = "xls/jit/aot_compiler_main"


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  out_base = pathlib.Path(_OUT_DIR.value)
  if not out_base.exists():
    out_base.mkdir()
  if not out_base.is_dir():
    raise app.UsageError(
        f"{_OUT_DIR.value} already exists and is not a directory"
    )
  llvm_ir = str(out_base / "result.ll")
  opt_ir = str(out_base / "result.opt.ll")
  asm = str(out_base / "result.asm")
  obj = str(out_base / "result.o")
  text_proto = str(out_base / "result.entrypoints.txtpb")
  proto_file = out_base / "result.entrypoints.pb"
  proto = str(proto_file)
  main_file = str(out_base / "main.cc")
  main_ll = str(out_base / "main.ll")
  linked_ll = str(out_base / "linked.ll")
  linked_opt_ll = str(out_base / "linked.opt.ll")
  top_arg = ["-top", _TOP.value] if _TOP.value else []
  mask_arg = ["-mask", _MASK.value] if _MASK.value else []
  print("Generating XLS artifacts")
  subprocess.run(
      [
          runfiles.get_path(_AOT_COMPILER),
          "-input",
          _IR.value,
          "-output_object",
          obj,
          "-output_proto",
          proto,
          "-output_textproto",
          text_proto,
          "-output_llvm_ir",
          llvm_ir,
          "-output_llvm_opt_ir",
          opt_ir,
          "-output_asm",
          asm,
      ]
      + top_arg,
      check=True,
  )
  with proto_file.open("rb") as entrypoints_file:
    entrypoints = aot_entrypoint_pb2.AotPackageEntrypointsProto.FromString(
        entrypoints_file.read()
    )
    if len(entrypoints.entrypoint) == 1:
      top_entrypoint = entrypoints.entrypoint[0]
    elif _TOP.value:

      def is_target(a: aot_entrypoint_pb2.AotEntrypointProto) -> bool:
        return (
            _TOP.value == a.xls_function_identifier
            or f"__{a.xls_package_name}__{_TOP.value}"
            == a.xls_function_identifier
        )

      poss = [a for a in entrypoints.entrypoint if is_target(a)]
      if len(poss) != 1:
        raise app.UsageError(f"Multiple possible entrypoints: {poss}")
      else:
        top_entrypoint = poss[0]
    else:
      raise app.UsageError("no tops found.")
  if top_entrypoint.type != aot_entrypoint_pb2.AotEntrypointProto.FUNCTION:
    print("Unable to generate main for non-function.")
    return
  if not _RESULT.value and not _INPUT.value:
    print("Can't generate a test-main without an input & result")
    return
  print("Generating main.cc")
  subprocess.run(
      [
          runfiles.get_path(_DUMP_MAIN),
          "-output",
          main_file,
          "-package",
          proto,
          "-result",
          _RESULT.value,
          "--write_result" if _WRITE_RESULT.value else "--nowrite_result",
      ]
      + mask_arg
      + top_arg
      + list(
          itertools.chain.from_iterable(["-input", i] for i in _INPUT.value)
      ),
      check=True,
  )
  if _CLANG.value:
    print("Compiling main.cc")
    subprocess.run(
        [
            _CLANG.value,
            "-S",
            "-emit-llvm",
            main_file,
            "-o",
            main_ll,
        ],
        check=True,
    )
    if _LLVM_LINK.value:
      print("Linking unopt main.cc")
      subprocess.run(
          [
              _LLVM_LINK.value,
              main_ll,
              llvm_ir,
              "-S",
              "-o",
              linked_ll,
          ],
          check=True,
      )
      print("Linking opt main.cc")
      subprocess.run(
          [
              _LLVM_LINK.value,
              main_ll,
              opt_ir,
              "-S",
              "-o",
              linked_opt_ll,
          ],
          check=True,
      )


if __name__ == "__main__":
  app.run(main)
