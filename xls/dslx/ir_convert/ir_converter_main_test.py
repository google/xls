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


_32BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=32
)
_128BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=128
)
_8BITS = xls_type_pb2.TypeProto(
    type_enum=xls_type_pb2.TypeProto.BITS, bit_count=8
)
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


def _proc(name: str, state: Iterable[_Param]):
  return xls_ir_interface_pb2.PackageInterfaceProto.Proc(
      base=xls_ir_interface_pb2.PackageInterfaceProto.FunctionBase(name=name),
      state=state,
  )


def _function(
    name: str,
    ret: xls_type_pb2.TypeProto,
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
      out.check_returncode()
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


if __name__ == "__main__":
  test_base.main()
