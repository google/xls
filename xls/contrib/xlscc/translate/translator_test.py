# Lint as: python3
#
# Copyright 2020 Google LLC
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

# pylint: disable=no-member
# pylint: disable=eval-used
# Disable no member error, linter seems confused.
# Disable eval warning. This is just a test.
"""Tests for xls.contrib.xlscc.translate.translator."""

import ctypes
import os
import re

import pycparser

from absl.testing import absltest
from xls.common import runfiles
from xls.contrib.xlscc.parse import ext_c_parser
from xls.contrib.xlscc.translate import hls_types_pb2
import xls.contrib.xlscc.translate.translator as xlscc_translator
from xls.ir.python import bits as bits_mod
from xls.ir.python import ir_interpreter
from xls.ir.python import value as ir_value


class TranslatorTest(absltest.TestCase):

  def parse_and_get_function(self, source, hls_types_by_name=None):
    hls_types_by_name = hls_types_by_name if hls_types_by_name else {}
    translator = xlscc_translator.Translator("mypackage", hls_types_by_name)
    my_parser = ext_c_parser.XLSccParser()
    my_parser.is_intrinsic_type = translator.is_intrinsic_type
    ast = my_parser.parse(source, "test_input")
    translator.parse(ast)
    p = translator.gen_ir()
    return p.get_function("test")

  def one_in_one_out(self, source, a_input, b_input, expected_output):
    f = self.parse_and_get_function("""
      int test(sai32 a, sai32 b) {
        """ + source + """
      }
    """)
    aval = ir_value.Value(bits_mod.SBits(value=int(a_input), bit_count=32))
    bval = ir_value.Value(bits_mod.SBits(value=int(b_input), bit_count=32))
    args = dict(a=aval, b=bval)
    result = ir_interpreter.run_function_kwargs(f, args)
    self.assertEqual(expected_output,
                     int(ctypes.c_int32(int(str(result))).value))

  def test_ref_params(self):
    f = self.parse_and_get_function("""
      void add_in_place(int &o, int x) {
        o += x;
      }
      int add_in_place_2(int x, int &o) {
        o += x;
        return o*2;
      }
      int test(int a, int b){
        add_in_place(a, b);
        return add_in_place_2(b, a) + a;
      }
    """)
    aval = ir_value.Value(bits_mod.SBits(value=int(11), bit_count=32))
    bval = ir_value.Value(bits_mod.SBits(value=int(3), bit_count=32))
    args = dict(a=aval, b=bval)
    result = ir_interpreter.run_function_kwargs(f, args)
    result_int = int(ctypes.c_int32(int(str(result))).value)
    self.assertEqual(51, result_int)

  def test_simple_add(self):
    self.one_in_one_out("return a+b;", 5, 1, 6)

  def test_simple_sign_extend(self):
    self.one_in_one_out("return uai3(a) + uai3(b);", 7, 2, 9)
    self.one_in_one_out("return sai3(a) + sai3(b);", 4, 3, -1)
    self.one_in_one_out("return uai3(a) + sai61(b);", 7, -1, 6)

  def test_simple_sub(self):
    self.one_in_one_out("return a-b;", 5, 1, 4)

  def test_simple_mul(self):
    self.one_in_one_out("return a*b;", 5, 5, 25)
    self.one_in_one_out("return a*b;", 5, -6, -30)

  def test_simple_shl(self):
    self.one_in_one_out("return a<<b;", 5, 5, 160)

  def test_simple_shr(self):
    self.one_in_one_out("return a>>b;", 256, 2, 64)

  def test_simple_and(self):
    self.one_in_one_out("return a&b;", 0b110101, 0b111, 0b101)

  def test_simple_or(self):
    self.one_in_one_out("return a|b;", 0b110101, 0b111, 0b110111)

  def test_simple_xor(self):
    self.one_in_one_out("return a^b;", 0b110101, 0b111111, 0b001010)

  def test_simple_bnot(self):
    self.one_in_one_out("return ~a;", 5, 0, -6)

  def test_simple_not_p(self):
    self.one_in_one_out("return !a;", 5, 0, 0)

  def test_simple_not_n(self):
    self.one_in_one_out("return !a;", 0, 0, 1)

  def test_simple_minus(self):
    self.one_in_one_out("return -a;", 5, 0, -5)

  def test_simple_less(self):
    self.one_in_one_out("return a<b;", 5, 6, 1)
    self.one_in_one_out("return a<b;", 5, 5, 0)

  def test_simple_lesseq(self):
    self.one_in_one_out("return a<=b;", 5, 6, 1)
    self.one_in_one_out("return a<=b;", 5, 5, 1)

  def test_simple_grt(self):
    self.one_in_one_out("return a>b;", 5, 4, 1)
    self.one_in_one_out("return a>b;", 5, 5, 0)

  def test_simple_grteq(self):
    self.one_in_one_out("return a>=b;", 5, 4, 1)
    self.one_in_one_out("return a>=b;", 5, 5, 1)

  def test_simple_eq(self):
    self.one_in_one_out("return a==b;", 5, 4, 0)
    self.one_in_one_out("return a==b;", 5, 5, 1)

  def test_simple_neq(self):
    self.one_in_one_out("return a!=b;", 5, 4, 1)
    self.one_in_one_out("return a!=b;", 5, 5, 0)

  def test_simple_lor(self):
    self.one_in_one_out("return a||b;", 5, 0, 1)
    self.one_in_one_out("return a||b;", 0, 0, 0)

  def test_simple_land(self):
    self.one_in_one_out("return a&&b;", 5, 0, 0)
    self.one_in_one_out("return a&&b;", 5, 3, 1)

  def test_simple_ternary(self):
    self.one_in_one_out("return a ? 3 : b;", 0, 11, 11)
    self.one_in_one_out("return a ? 3 : b;", 1, 11, 3)

  def test_simple_inc(self):
    self.one_in_one_out("int x = ++a; return x+a;", 5, 0, 12)

  def test_simple_post_inc(self):
    self.one_in_one_out("int x = a++; return x+a;", 5, 0, 11)

  def test_simple_dec(self):
    self.one_in_one_out("int x = --a; return x+a;", 5, 0, 8)

  def test_simple_post_dec(self):
    self.one_in_one_out("int x = a--; return x+a;", 5, 0, 9)

  def test_simple_bool(self):
    self.one_in_one_out("return (a==55) && (b==60);", 55, 60, 1)
    self.one_in_one_out("return bool(a) == bool(b);", 0, 60, 0)

  def test_simple_slc(self):
    # Sign extension
    self.one_in_one_out("return a.slc<3>(2);", 0b1001011001, 0, -2)

  def test_simple_set_slc(self):
    self.one_in_one_out("a.set_slc(1, uai2(3));return a;", 0b0001011001, 0, 95)

  def test_simple_cpp_cast(self):
    self.one_in_one_out("return uai5(a);", 55, 0, 23)

  def test_enum_autocount(self):
    source = """
    enum states{w, x, y=5, z};
    int test(int a, int b){
        return a+z+b+x;
    }
    """
    f = self.parse_and_get_function(source)
    aval = ir_value.Value(bits_mod.SBits(value=int(2), bit_count=32))
    bval = ir_value.Value(bits_mod.SBits(value=int(3), bit_count=32))
    args = dict(a=aval, b=bval)
    result = ir_interpreter.run_function_kwargs(f, args)
    result_int = int(ctypes.c_int32(int(str(result))).value)
    self.assertEqual(12, result_int)

  def test_simple_switch(self):
    source = """
    switch(a) {
      default:
      case 0:
        return b;
      case 1:
        return a;
      case 5:
        return a+b;
    }
    """
    self.one_in_one_out(source, 0, 222, 222)
    self.one_in_one_out(source, 1, 222, 1)
    self.one_in_one_out(source, 2, 222, 222)
    self.one_in_one_out(source, 5, 222, 227)

  def test_simple_unrolled_loop(self):
    source = """
    int ret = 0;
    #pragma hls_unroll yes
    for(int i=0;i<10;++i) {
      if(i==1) {
        ret -= b;
        continue;
      }
      ret += a;
      if(i==3)
        break;
    }
    return ret;
    """
    self.one_in_one_out(source, 10, 3, 27)

    # TODO(seanhaskell): Broken parsing
#  def test_simple_structref_cast(self):
#    self.one_in_one_out("return uai32(a).slc<3>(2);", -5, 0, -11)
#    self.one_in_one_out("return ((uai32)a).slc<3>(2);", -5, 0, 6)
  
  def test_invalid_struct_operation(self):
      statestruct = hls_types_pb2.HLSStructType()

      int_type = hls_types_pb2.HLSIntType()
      int_type.signed = True
      int_type.width = 18

      translated_hls_type = hls_types_pb2.HLSType()
      translated_hls_type.as_int.CopyFrom(int_type)

      statestructtype = hls_types_pb2.HLSType()
      statestructtype.as_struct.CopyFrom(statestruct)
      hls_types_by_name = {"State": statestructtype}

      source = """
      enum states{State_TX=1};
      int test(){
        int ret = 0;
        State state;
        ret = state - 1;
        return ret;
      }
      """
      with self.assertRaisesRegex(ValueError, 'Invalid binary operand at'):
          f = self.parse_and_get_function(source, hls_types_by_name)

  def test_invalid_comparison(self):
      statestruct = hls_types_pb2.HLSStructType()

      int_type = hls_types_pb2.HLSIntType()
      int_type.signed = True
      int_type.width = 18

      translated_hls_type = hls_types_pb2.HLSType()
      translated_hls_type.as_int.CopyFrom(int_type)

      statestructtype = hls_types_pb2.HLSType()
      statestructtype.as_struct.CopyFrom(statestruct)
      hls_types_by_name = {"State": statestructtype}

      source = """
      enum states{State_TX=1};
      int test(){
        int ret = 0;
        State state;
        if (state == State_TX){
            return State_TX;
        }
        return ret;
      }
      """
      with self.assertRaisesRegex(ValueError, 'Invalid operand at'):
          f = self.parse_and_get_function(source, hls_types_by_name)

  def test_structref(self):
    somestruct = hls_types_pb2.HLSStructType()

    int_type = hls_types_pb2.HLSIntType()
    int_type.signed = True
    int_type.width = 18

    translated_hls_type = hls_types_pb2.HLSType()
    translated_hls_type.as_int.CopyFrom(int_type)

    hls_field = hls_types_pb2.HLSNamedType()
    hls_field.name = "x"
    hls_field.hls_type.CopyFrom(translated_hls_type)
    somestruct.fields.add().CopyFrom(hls_field)

    hls_field = hls_types_pb2.HLSNamedType()
    hls_field.name = "y"
    hls_field.hls_type.CopyFrom(translated_hls_type)
    somestruct.fields.add().CopyFrom(hls_field)

    somestructtype = hls_types_pb2.HLSType()
    somestructtype.as_struct.CopyFrom(somestruct)
    hls_types_by_name = {"SomeStruct": somestructtype}

    f = self.parse_and_get_function("""
      void struct_update(SomeStruct &o, sai18 x) {
        o.x.set_slc(0, x);
      }
      int test(sai32 a, int b) {
        SomeStruct s;
        s.x = 0;
        s.y = b;
        struct_update(s, a);
        return s.x + s.y;
      }
    """, hls_types_by_name)
    aval = ir_value.Value(bits_mod.SBits(value=int(22), bit_count=32))
    bval = ir_value.Value(bits_mod.SBits(value=int(10), bit_count=32))
    args = dict(a=aval, b=bval)
    result = ir_interpreter.run_function_kwargs(f, args)
    result_int = int(ctypes.c_int32(int(str(result))).value)
    self.assertEqual(32, result_int)

  def test_arrayref(self):
    f = self.parse_and_get_function("""
      void array_update(int o[2], int x) {
        o[0] = x;
      }
      int test(int a, int b) {
        int s[2] = {0,b};
        array_update(s, a);
        return s[0] + s[1];
      }
    """)
    aval = ir_value.Value(bits_mod.SBits(value=int(22), bit_count=32))
    bval = ir_value.Value(bits_mod.SBits(value=int(10), bit_count=32))
    args = dict(a=aval, b=bval)
    result = ir_interpreter.run_function_kwargs(f, args)
    result_int = int(ctypes.c_int32(int(str(result))).value)
    self.assertEqual(32, result_int)

  def test_globalconst(self):
    f = self.parse_and_get_function("""
      const int foo[6] = {2,4,5,3,2,1};
      int test(int a) {
        return foo[a];
      }
    """)
    aval = ir_value.Value(bits_mod.SBits(value=int(2), bit_count=32))
    args = dict(a=aval)
    result = ir_interpreter.run_function_kwargs(f, args)
    result_int = int(ctypes.c_int32(int(str(result))).value)
    self.assertEqual(5, result_int)

  def test_mandelbrot(self):
    translator = xlscc_translator.Translator("mypackage")

    my_parser = ext_c_parser.XLSccParser()

    source_path = runfiles.get_path(
        "xls/contrib/xlscc/translate/testdata/mandelbrot_test.cc")
    binary_path = runfiles.get_path(
        "xls/contrib/xlscc/translate/mandelbrot_test")

    nx = 48
    ny = 32

    with open(source_path) as f:
      content = f.read()

    # Hackily do the preprocessing, since in build environments we do not have a
    # cpp binary available (and if we did relying on it here without a
    # build-system-noted dependency would be non-hermetic).
    content = re.sub("^#if !NO_TESTBENCH$.*^#endif$", "", content, 0,
                     re.MULTILINE | re.DOTALL)
    content = re.sub(re.compile("//.*?\n"), "", content)

    f = self.create_tempfile(content=content)
    ast = pycparser.parse_file(f.full_path, use_cpp=False, parser=my_parser)

    cpp_out = None
    with os.popen(binary_path) as osf:
      cpp_out = osf.read()
    parsed_cpp_out = eval(cpp_out)

    translator.parse(ast)

    p = translator.gen_ir()
    f = p.get_function("mandelbrot")

    result_arr = parsed_cpp_out

    for y in range(0, ny):
      for x in range(0, nx):
        xx = float(x) / nx
        yy = float(y) / ny

        xi = int(xx * 2.5 - 1.8) * (1<<16)
        yi = int(yy * 2.2 - 1.1) * (1<<16)

        args = dict(c_r=ir_value.Value(bits_mod.SBits(value=int(xi),
                                                      bit_count=32)),
                    c_i=ir_value.Value(bits_mod.SBits(value=int(yi),
                                                      bit_count=32)))

        result = ir_interpreter.run_function_kwargs(f, args)

        result_sai32 = int(str(result))
        result_arr[y][x] = result_sai32

    self.assertEqual(parsed_cpp_out, result_arr)


if __name__ == "__main__":
  absltest.main()
