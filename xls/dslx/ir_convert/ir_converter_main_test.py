#
# Copyright 2021 The XLS Authors
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
"""Tests for ir_converter_main."""

import dataclasses
import os
import subprocess
import tempfile
import textwrap
from typing import Dict, Iterable, Optional

from xls.common import runfiles
from xls.common import test_base
from xls.ir import xls_ir_interface_pb2
from xls.ir import xls_type_pb2


@dataclasses.dataclass
class ConvertResult:
  """Result of running the ir_converter_main binary."""

  ir: str
  stderr: str
  interface: xls_ir_interface_pb2.PackageInterfaceProto


@dataclasses.dataclass(frozen=True)
class SrcFile:
  name: str

  @property
  def path(self):
    return runfiles.get_path(
        f"xls/dslx/ir_convert/testdata/sv_types/{self.name}"
    )


_SV_DECL_1 = SrcFile("sv_declarations_one.x")


_TOKEN = xls_type_pb2.TypeProto(type_enum=xls_type_pb2.TypeProto.TOKEN)
_32BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=32
)
_128BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=128
)
_8BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=8
)
_1BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=1
)
_EMPTY_TUPLE = xls_type_pb2.TypeProto(type_enum=xls_type_pb2.TypeProto.TUPLE)
_MY_STRUCT = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.TUPLE, tuple_elements=[_128BITS, _8BITS]
)
_MY_ENUM = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=2
)
_MY_TUPLE = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.TUPLE,
    tuple_elements=[_8BITS, _8BITS, _8BITS],
)

_Package = xls_ir_interface_pb2.PackageInterfaceProto
_Param = xls_ir_interface_pb2.PackageInterfaceProto.NamedValue
_Channel = xls_ir_interface_pb2.PackageInterfaceProto.Channel


def _proc(name: str, state: Iterable[_Param], top: Optional[bool] = None):
  return xls_ir_interface_pb2.PackageInterfaceProto.Proc(
      base=xls_ir_interface_pb2.PackageInterfaceProto.FunctionBase(
          name=name, top=top
      ),
      state=state,
  )


def _function(
    name: str,
    ret: Optional[xls_type_pb2.TypeProto] = None,
    top: Optional[bool] = None,
    args: Optional[
        Iterable[xls_ir_interface_pb2.PackageInterfaceProto.NamedValue]
    ] = None,
    sv_ret: Optional[str] = None,
) -> xls_ir_interface_pb2.PackageInterfaceProto.Function:
  return xls_ir_interface_pb2.PackageInterfaceProto.Function(
      parameters=args,
      result_type=ret,
      sv_result_type=sv_ret,
      base=xls_ir_interface_pb2.PackageInterfaceProto.FunctionBase(
          top=top, name=name
      ),
  )


class IrConverterMainTest(test_base.TestCase):
  A_DOT_X = "fn f() -> u32 { u32:42 }"
  B_DOT_X = "fn f() -> u32 { u32:64 }"
  IR_CONVERTER_MAIN_PATH = runfiles.get_path(
      "xls/dslx/ir_convert/ir_converter_main"
  )

  def _ir_convert_files(
      self,
      dslx: Iterable[SrcFile],
      package_name: Optional[str] = None,
      *,
      extra_flags: Iterable[str] = (),
      expect_zero_exit: bool = True,
  ):
    interface = self.create_tempfile()
    cmd = [
        self.IR_CONVERTER_MAIN_PATH,
        f"--interface_proto_file={interface.full_path}",
        f"--dslx_path={os.path.dirname(_SV_DECL_1.path)}",
    ] + [s.path for s in dslx]
    if package_name is not None:
      cmd.append("--package_name=" + package_name)
    cmd.extend(extra_flags)
    out = subprocess.run(
        cmd,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_zero_exit:
      out.check_returncode()
    return ConvertResult(
        ir=out.stdout,
        stderr=out.stderr,
        interface=xls_ir_interface_pb2.PackageInterfaceProto.FromString(
            interface.read_bytes()
        ),
    )

  def _ir_convert(
      self,
      dslx_contents: Dict[str, str],
      package_name: Optional[str] = None,
      *,
      extra_flags: Iterable[str] = (),
      expect_zero_exit: bool = True,
      convert_tests: bool = False,
      top: Optional[str] = None,
  ) -> ConvertResult:
    interface = self.create_tempfile()
    tempdir = self.create_tempdir()
    tempfiles = []
    for filename, contents in dslx_contents.items():
      path = os.path.join(tempdir, filename)
      with open(path, "w") as f:
        f.write(contents)
      tempfiles.append(filename)
    cmd = [
        self.IR_CONVERTER_MAIN_PATH,
        f"--interface_proto_file={interface.full_path}",
    ] + tempfiles
    if convert_tests:
      cmd.append("--convert_tests")
    if top is not None:
      cmd.append(f"--top={top}")
    if package_name is not None:
      cmd.append("--package_name=" + package_name)
    cmd.extend(extra_flags)
    out = subprocess.run(
        cmd,
        encoding="utf-8",
        cwd=tempdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_zero_exit:
      try:
        out.check_returncode()
      except Exception as e:
        e.add_note(f"stdout: {out.stdout}\nstderr: {out.stderr}")
        raise e
    return ConvertResult(
        ir=out.stdout,
        stderr=out.stderr,
        interface=xls_ir_interface_pb2.PackageInterfaceProto.FromString(
            interface.read_bytes()
        ),
    )

  def test_bad_package_name(self) -> None:
    result = self._ir_convert(
        {"a-name-with-minuses.x": self.A_DOT_X},
        expect_zero_exit=False,
    )
    self.assertEmpty(result.ir)
    self.assertRegex(
        result.stderr,
        r"package name 'a-name-with-minuses' \(len: 19\) is not a valid package"
        r" name",
    )

  def test_bad_package_name_given(self) -> None:
    result = self._ir_convert(
        {"foo.x": self.A_DOT_X},
        package_name="a-name-with-minuses",
        expect_zero_exit=False,
    )
    self.assertEmpty(result.ir)
    self.assertRegex(
        result.stderr,
        r"package name 'a-name-with-minuses' \(len: 19\) is not a valid package"
        r" name",
    )

  def test_a_dot_x(self) -> None:
    result = self._ir_convert({"a.x": self.A_DOT_X})
    self.assertEqual(
        result.ir,
        textwrap.dedent("""\
    package a

    file_number 0 "a.x"

    fn __a__f() -> bits[32] {
      ret literal.1: bits[32] = literal(value=42, id=1, pos=[(0,0,16)])
    }
    """),
    )
    self.assertEqual(
        result.interface,
        _Package(
            name="a",
            files=["a.x"],
            functions=[_function(name="__a__f", ret=_32BITS)],
        ),
    )

  def test_b_dot_x(self) -> None:
    self.assertEqual(
        self._ir_convert({"b.x": self.B_DOT_X}).ir,
        textwrap.dedent("""\
    package b

    file_number 0 "b.x"

    fn __b__f() -> bits[32] {
      ret literal.1: bits[32] = literal(value=64, id=1, pos=[(0,0,16)])
    }
    """),
    )

  def test_multi_file(self) -> None:
    self.assertEqual(
        self._ir_convert(
            {"a.x": self.A_DOT_X, "b.x": self.B_DOT_X}, package_name="my_entry"
        ).ir,
        textwrap.dedent("""\
    package my_entry

    file_number 0 "a.x"
    file_number 1 "b.x"

    fn __a__f() -> bits[32] {
      ret literal.1: bits[32] = literal(value=42, id=1, pos=[(0,0,16)])
    }

    fn __b__f() -> bits[32] {
      ret literal.2: bits[32] = literal(value=64, id=2, pos=[(1,0,16)])
    }
    """),
    )

  def test_execute_sv_declarations_one_my_tuple(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_one_my_tuple.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_one_my_tuple__execute",
                    ret=_MY_TUPLE,
                    args=[_Param(name="i", type=_MY_TUPLE, sv_type="sv_tuple")],
                    sv_ret="sv_tuple",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_one_non_sv_tuple(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_one_non_sv_tuple.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_one_non_sv_tuple__execute",
                    ret=_MY_TUPLE,
                    args=[_Param(name="i", type=_MY_TUPLE, sv_type=None)],
                    sv_ret=None,
                )
            ],
        ),
    )

  def test_execute_sv_declarations_one_my_struct(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_one_my_struct.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_one_my_struct__execute",
                    ret=_MY_STRUCT,
                    args=[
                        _Param(name="i", type=_MY_STRUCT, sv_type="sv_struct")
                    ],
                    sv_ret="sv_struct",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_one_my_enum(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_one_my_enum.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_one_my_enum__execute",
                    ret=_MY_ENUM,
                    args=[_Param(name="i", type=_MY_ENUM, sv_type="sv_enum")],
                    sv_ret="sv_enum",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_re_export_my_tuple(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_re_export_my_tuple.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_two_re_export_my_tuple__execute",
                    ret=_MY_TUPLE,
                    args=[_Param(name="i", type=_MY_TUPLE, sv_type="sv_tuple")],
                    sv_ret="sv_tuple",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_rename_my_tuple(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_rename_my_tuple.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name=(
                        "__execute_sv_declarations_two_rename_my_tuple__execute"
                    ),
                    ret=_MY_TUPLE,
                    args=[
                        _Param(
                            name="i", type=_MY_TUPLE, sv_type="rename_sv_tuple"
                        )
                    ],
                    sv_ret="rename_sv_tuple",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_add_name_tuple(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_add_name_tuple.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name=(
                        "__execute_sv_declarations_two_add_name_tuple__execute"
                    ),
                    ret=_MY_TUPLE,
                    args=[
                        _Param(name="i", type=_MY_TUPLE, sv_type="new_sv_tuple")
                    ],
                    sv_ret="new_sv_tuple",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_rename_my_struct(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_rename_my_struct.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_two_rename_my_struct__execute",
                    ret=_MY_STRUCT,
                    args=[
                        _Param(
                            name="i",
                            type=_MY_STRUCT,
                            sv_type="rename_sv_struct",
                        )
                    ],
                    sv_ret="rename_sv_struct",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_re_export_my_struct(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_re_export_my_struct.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_two_re_export_my_struct__execute",
                    ret=_MY_STRUCT,
                    args=[
                        _Param(name="i", type=_MY_STRUCT, sv_type="sv_struct")
                    ],
                    sv_ret="sv_struct",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_re_export_my_enum(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_re_export_my_enum.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name="__execute_sv_declarations_two_re_export_my_enum__execute",
                    ret=_MY_ENUM,
                    args=[_Param(name="i", type=_MY_ENUM, sv_type="sv_enum")],
                    sv_ret="sv_enum",
                )
            ],
        ),
    )

  def test_execute_sv_declarations_two_rename_my_enum(self) -> None:
    result = self._ir_convert_files(
        [
            SrcFile("execute_sv_declarations_two_rename_my_enum.x"),
        ],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            functions=[
                _function(
                    name=(
                        "__execute_sv_declarations_two_rename_my_enum__execute"
                    ),
                    ret=_MY_ENUM,
                    args=[
                        _Param(
                            name="i", type=_MY_ENUM, sv_type="rename_sv_enum"
                        )
                    ],
                    sv_ret="rename_sv_enum",
                )
            ],
        ),
    )

  def test_proc_channel(self) -> None:
    result = self._ir_convert_files(
        [SrcFile("proc_channel_test.x")],
        "exec",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="exec",
            channels=[
                _Channel(
                    name="exec__foo",
                    type=_MY_TUPLE,
                    direction=xls_ir_interface_pb2.PackageInterfaceProto.Channel.IN,
                    sv_type="sv_tuple",
                ),
                _Channel(
                    name="exec__bar",
                    type=_MY_TUPLE,
                    direction=xls_ir_interface_pb2.PackageInterfaceProto.Channel.OUT,
                    sv_type="sv_tuple",
                ),
            ],
            procs=[
                _proc(
                    name="__proc_channel_test__Execute_0_next",
                    state=[
                        _Param(
                            name="__state",
                            type=xls_type_pb2.TypeProto(
                                type_enum=xls_type_pb2.TypeProto.TUPLE
                            ),
                        )
                    ],
                )
            ],
        ),
    )

  PROC_TEST_DOT_X = """
    proc DeepThought {
        start: chan<()> in;
        answer: chan<u32> out;
        init { () }
        config (start: chan<()> in, answer: chan<u32> out) {
            (start, answer)
        }
        next(st: ()) {
            let (tok, _) = recv(join(), start);
            send(tok, answer, u32:42);
        }
    }
    #[test_proc]
    proc TestPass {
        terminator: chan<bool> out;
        start_s: chan<()> out;
        result_r: chan<u32> in;
        init { () }
        config (terminator: chan<bool> out) {
            let (start_s, start_r) = chan<()>("start_chan");
            let (result_s, result_r) = chan<u32>("result_chan");
            spawn DeepThought(start_r, result_s);
            (terminator, start_s, result_r)
        }
        next(state: ()) {
            let tok = send(join(), start_s, ());
            let (tok, res) = recv(join(), result_r);
            assert_eq(res, u32:42);
            send(tok, terminator, true);
        }
    }
    #[test_proc]
    proc TestFail {
        terminator: chan<bool> out;
        start_s: chan<()> out;
        result_r: chan<u32> in;
        init { () }
        config (terminator: chan<bool> out) {
            let (start_s, start_r) = chan<()>("start_chan");
            let (result_s, result_r) = chan<u32>("result_chan");
            spawn DeepThought(start_r, result_s);
            (terminator, start_s, result_r)
        }
        next(state: ()) {
            let tok = send(join(), start_s, ());
            let (tok, res) = recv(join(), result_r);
            assert_eq(res, u32:64);
            send(tok, terminator, true);
        }
    }
    """

  def test_convert_test_proc(self):
    result = self._ir_convert(
        {"d.x": self.PROC_TEST_DOT_X},
        convert_tests=True,
        package_name="foo",
        top="TestPass",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="foo",
            channels=[
                _Channel(
                    name="foo__terminator",
                    type=_1BITS,
                    direction=xls_ir_interface_pb2.PackageInterfaceProto.Channel.OUT,
                )
            ],
            procs=[
                _proc(
                    name="__d__TestPass_0_next",
                    top=True,
                    state=[_Param(name="__state", type=_EMPTY_TUPLE)],
                ),
                _proc(
                    name="__d__TestPass__DeepThought_0_next",
                    state=[_Param(name="__state", type=_EMPTY_TUPLE)],
                ),
            ],
            functions=[
                _function(
                    name="__d__DeepThought.init",
                    ret=_EMPTY_TUPLE,
                ),
            ],
        ),
    )

  FN_TEST_DOT_X = """
    fn answer() -> u32 { u32:42 }
    #[test]fn test_pass() { assert_eq(answer(), u32:42); }
    #[test]fn test_fail() { assert_eq(answer(), u32:64); }
    """

  def test_convert_test_fn(self):
    result = self._ir_convert(
        {"c.x": self.FN_TEST_DOT_X},
        convert_tests=True,
        package_name="foo",
        top="test_pass",
    )
    # file names depend on where runfiles are put.
    del result.interface.files[:]
    self.assertEqual(
        result.interface,
        _Package(
            name="foo",
            functions=[
                _function(
                    name="__c__answer",
                    ret=_32BITS,
                ),
                _function(
                    top=True,
                    name="__itok__c__test_pass",
                    ret=xls_type_pb2.TypeProto(
                        type_enum=xls_type_pb2.TypeProto.TUPLE,
                        tuple_elements=[_TOKEN, _EMPTY_TUPLE],
                    ),
                    args=[
                        _Param(name="__token", type=_TOKEN),
                        _Param(name="__activated", type=_1BITS),
                    ],
                ),
                _function(
                    name="__c__test_pass",
                ),
            ],
        ),
    )

  def test_alternative_stdlib_path(self):
    with tempfile.TemporaryDirectory(suffix="stdlib") as stdlib_dir:
      # Make a std.x file in our fake stdlib.
      with open(os.path.join(stdlib_dir, "std.x"), "w") as fake_std:
        print("pub fn my_stdlib_func(x: u32) -> u32 { x }", file=fake_std)

      # Invoke the function in our fake std.x which should be appropriately
      # resolved via the dslx_stdlib_path flag.
      program = """
      import std;

      fn main() -> u32 { std::my_stdlib_func(u32:42) }
      """
      result = self._ir_convert(
          {"main.x": program},
          extra_flags=[f"--dslx_stdlib_path={stdlib_dir}"],
      )
      self.assertIn(
          textwrap.dedent("""\
      fn __std__my_stdlib_func(x: bits[32] id=1) -> bits[32] {
        ret x: bits[32] = param(name=x, id=1)
      }
      """),
          result.ir,
      )


if __name__ == "__main__":
  test_base.main()
