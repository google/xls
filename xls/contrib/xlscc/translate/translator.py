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
"""Translates a subset of C++ to XLS IR."""

import collections
import copy
import re

from xls.contrib.xlscc.parse import ext_c_ast as c_ast
from xls.contrib.xlscc.translate import hls_types_pb2
from xls.ir.python import bits as bits_mod
from xls.ir.python import fileno as ir_fileno
from xls.ir.python import function_builder
from xls.ir.python import package as ir_package
from xls.ir.python import source_location
from xls.ir.python import value as ir_value


class Type(object):
  """Base class for C++ HLS Types.
  """

  def __init__(self):
    self.bit_width = 0


class IntType(Type):
  """Class for C++ HLS Integer Types.
  """

  def __init__(self, width, signed, native):
    super(IntType, self).__init__()
    self.bit_width = width
    self.signed = signed
    self.native = native

  def get_xls_type(self, p):
    return p.get_bits_type(self.bit_width)

  def __str__(self):
    if self.native:
      pre = "" if self.signed else "unsigned_"
      if self.bit_width == 32:
        return pre + "int"
      elif self.bit_width == 1:
        return pre + "psuedobool"
      elif self.bit_width == 64:
        return pre + "int64"
      elif self.bit_width == 16:
        return pre + "short"
      elif self.bit_width == 8:
        return pre + "char"
    return "sai" + str(self.bit_width) if self.signed else "uai" + str(
        self.bit_width)


class BoolType(Type):
  """Class for C++ HLS Boolean Types.
  """

  def __init__(self):
    super(BoolType, self).__init__()
    self.bit_width = 1

  def get_xls_type(self, p):
    return p.get_bits_type(1)

  def __str__(self):
    return "bool"


class VoidType(Type):
  """Class for C++ HLS Void Type.
  """

  def __str__(self):
    return "void"


class ChannelType(Type):
  """Class for C++ HLS channel Type."""

  def __init__(self, channel_type):
    super(ChannelType, self).__init__()
    self.channel_type = channel_type
    self.is_input = None

  def set_is_input(self, is_input):
    self.is_input = is_input

  def get_xls_type(self, p):
    """Get XLS IR type for channel.

    Args:
      p: XLS IR package

    Returns:
      XLS IR tuple of field types
    """
    assert self.is_input is not None
    if self.is_input:
      return p.get_tuple_type(
          [p.get_bits_type(1),
           self.channel_type.get_xls_type(p)])
    else:
      return p.get_bits_type(1)

  def __str__(self):
    return "channel<{t}>".format(t=self.channel_type)


class ChannelReadType(Type):
  """Class for C++ HLS channel Type."""

  def __init__(self, channel_type):
    super().__init__()
    self.channel_type = channel_type
    self.is_input = None

  def set_is_input(self, is_input):
    self.is_input = is_input

  def get_xls_type(self, p):  # pylint: disable=unused-argument
    # Use channel_type
    return None

  def __str__(self):
    return "channel<{t}>".format(t=self.channel_type)


class StructType(Type):
  """Class for C++ HLS Struct Types.

    Args:
      name: Name of the struct type
      struct: HLSStructType protobuf
  """

  def __init__(self, name, struct):
    
    super(StructType, self).__init__()
    self.name = name
    if isinstance(struct, c_ast.Struct):
      self.is_const = False
      self.struct = struct 
      self.as_struct = False
      self.field_indices = {}
      self.element_types = {}
      for decl in struct.decls:
        name = decl.name
        field = decl.type
        #print("Field is " + str(field))
        self.field_indices[name] = len(self.field_indices)
        if isinstance(field, c_ast.TypeDecl):
          if field.type.names is 'int':
              self.element_types[name] = IntType(32, True, True)
          elif field.type.names is 'struct':
              self.element_types[name] = StructType(field.type.names)
          assert isinstance(field.type, c_ast.IdentifierType)
        elif isinstance(field, c_ast.IdentifierType):
          print(field)
        else:
          raise NotImplementedError("Unsupported field type for field", name,
                                  ":", type(field))
    #print("Self struct is " + str(self.field_indices))

    else:
      self.struct = struct
      self.field_indices = {}
      self.element_types = {}
      for named_field in self.struct.fields:
        name = named_field.name
        field = named_field.hls_type
        self.field_indices[name] = len(self.field_indices)
        if field.HasField("as_int"):
          self.element_types[name] = IntType(field.as_int.width,
                                           field.as_int.signed,
                                           False)
        elif field.HasField("as_struct"):
          self.element_types[name] = StructType(name, field.as_struct)
        else:
          raise NotImplementedError("Unsupported field type for field", name,
                                  ":", type(field))
      self.bit_width = self.bit_width + self.element_types[name].bit_width

 
   
  def get_xls_type(self, p):
    """Get XLS IR type for struct.

    Args:
      p: XLS IR package
    Returns:
      XLS IR tuple of field types
    """

    element_xls_types = []
    for named_field in self.struct.fields:
      name = named_field.name
      field = named_field.hls_type
      if field.HasField("as_int"):
        element_xls_types.append(p.get_bits_type(field.as_int.width))
      elif field.HasField("as_struct"):
        element_xls_types.append(self.element_types[name].get_xls_type(p))
      elif field.HasField("as_array"):
        element_xls_types.append(self.element_types[name].get_xls_type(p))
      else:
        raise NotImplementedError("Unsupported struct field type in C AST",
                                  type(field))

    return p.get_tuple_type(element_xls_types)

  def get_field_indices(self):
    return self.field_indices

  def get_element_type(self, name):
    return self.element_types[name]

  def get_element_types(self):
    return self.element_types.values()

  def __str__(self):
    return self.name


class ArrayType(Type):
  """Class for C++ HLS Array Types.
  """

  def __init__(self, element_type, size):
    super(ArrayType, self).__init__()
    self.element_type = element_type
    self.size = size
    self.bit_width = self.element_type.bit_width * self.size

  def get_element_type(self):
    return self.element_type

  def get_size(self):
    return self.size

  def get_xls_type(self, p):
    return p.get_array_type(self.size, self.element_type.get_xls_type(p))

  def __str__(self):
    return "{type}[{size}]".format(type=self.element_type, size=self.size)


class CVar(object):
  """Represents a C RValue.

     Binds together the function builder value and C Type.
  """

  def __init__(self, fb_expr, ctype):
    self.fb_expr = fb_expr
    self.ctype = ctype


class Function(object):
  """Represents a C function.

     Binds together the XLS function builder value and C Type.
  """

  def parse_function(self, translator, ast):
    """Parses the function from C Ast.

    Args:
      translator: Translator object
      ast: pycparser ast
    """
    assert isinstance(ast, c_ast.FuncDef)
    decl = ast.decl
    self.name = decl.name

    self.loc = translate_loc(ast)

    func_decl = decl.type
    assert isinstance(func_decl, c_ast.FuncDecl)

    param_list = func_decl.args
    type_decl = func_decl.type

    # parse return type
    self.return_type = translator.parse_type(type_decl)

    # parse parameters
    self.params = collections.OrderedDict()
    if param_list is not None:
      for name, child in param_list.children():
        assert isinstance(child, c_ast.Decl)

        name = child.name

        t = translator.parse_type(child.type)
        if isinstance(t, ChannelType):
          # Only valid for the main function
          # TODO(seanhaskell): Support passing to helper functions
          assert (name in translator.channels_in) or (
              name in translator.channels_out)
          t.set_is_input(name in translator.channels_in)

        self.params[name] = t

    # parse body
    self.body_ast = ast.body

  def set_fb_expr(self, fb_expr):
    self.fb_expr = fb_expr

  def get_ref_param_names(self):

    def ref_name(name):
      param_type = self.params[name]
      return param_type.is_ref and (not param_type.is_const) and (
          not isinstance(param_type, ChannelType))

    return list(filter(ref_name, self.params))

  def get_ref_param_indices(self):
    ret = []
    idx = 0
    for _, param_type in self.params.items():
      if param_type.is_ref and (not param_type.is_const):
        ret.append(idx)
      idx = idx + 1

    return ret

  def count_returns(self):
    """Returns # of vals effectively returned (refparams, channels, retval)."""
    reference_param_names = self.get_ref_param_names()
    n_channel_returns = 0
    for _, ptype in self.params.items():
      if isinstance(ptype, ChannelType):
        n_channel_returns = n_channel_returns + 1
        if not ptype.is_input:
          n_channel_returns = n_channel_returns + 1
    if not isinstance(self.return_type, VoidType):
      return len(reference_param_names) + n_channel_returns + 1
    else:
      return len(reference_param_names) + n_channel_returns


def parse_constant(stmt_ast):
  """Parses a constant from the C++ AST.

  Args:
    stmt_ast: pycparser AST
  Returns:
    integer value, C Type
  """

  if stmt_ast.type == "int" or stmt_ast.type == "long":
    if stmt_ast.value[-3:] == "ull":
      const_type = IntType(64, True, True)
    else:
      const_type = IntType(32, True, True)
  elif stmt_ast.type == "long int":
    const_type = IntType(64, True, True)
  elif stmt_ast.type == "bool":
    name = stmt_ast.value
    if not ((name == "true") or (name == "false")):
      raise ValueError("Bool values must be true or false")
    return name == "true", BoolType()
  else:
    raise ValueError("Unknown constant type: ", stmt_ast.type)
  stripped = re.search("[0-9]+", stmt_ast.value).group(0)
  return int(stripped), const_type


def translate_loc(stmt_ast):
  """Translates a C source location from AST to XLS IR source location.

  Args:
    stmt_ast: pycparser AST
  Returns:
    XLS IR source location
  """
  if stmt_ast is None:
    return None
  if isinstance(stmt_ast, source_location.SourceLocation):
    return stmt_ast
  if stmt_ast.coord is None:
    return None
  return source_location.SourceLocation(
      ir_fileno.Fileno(0), ir_fileno.Lineno(stmt_ast.coord.line),
      ir_fileno.Colno(stmt_ast.coord.column))


class RetVal(object):
  """Represents a possible return value from a function.

  Combines RValue, type, and condition for returning this value.
  """

  def __init__(self, condition, fb_expr, return_type):
    self.condition = condition
    self.fb_expr = fb_expr
    self.return_type = return_type


class Translator(object):
  """Converts from C++ to XLS IR. Usage is first parse(ast), then gen_ir().
  """

  def __init__(self,
               package_name,
               hls_types_by_name=None,
               channels_in=None,
               channels_out=None):
    self.functions_ = {}
    self.global_decls_ = {}
    self.package_name_ = package_name
    self.hls_types_by_name_ = hls_types_by_name if hls_types_by_name else {}
    self.channels_in = channels_in
    self.channels_out = channels_out

  def is_intrinsic_type(self, name):
    if name.startswith("uai") or name.startswith("sai"):
      return True
    if name == "ac_int":
      return True
    if name == "ac_channel":
      return True
    if name in self.hls_types_by_name_:
      return True
    return False

  def get_struct_type(self, name):
    if name not in self.hls_types_by_name_:
      return None
    if not self.hls_types_by_name_[name].as_struct:
      return None
    return self.hls_types_by_name_[name].as_struct

  def parse(self, ast):
    """Parses a C source file's AST.

    Args:
      ast: pycparser AST
    """
    for name, child in ast.children():
      if isinstance(child, c_ast.FuncDef):
        func = Function()
        func.parse_function(self, child)
        self.functions_[func.name] = func
      elif isinstance(child, c_ast.Decl):
        name = child.name
        assert name not in self.global_decls_
        if isinstance(child.type, c_ast.ArrayDecl):
          if child.quals == ["const"]:
            elem_type = self.parse_type(child.type.type)
            assert isinstance(child.init, c_ast.InitList)
            values = []
            for elem_ast in child.init.exprs:
              assert isinstance(elem_ast, c_ast.Constant)
              values.append(
                  ir_value.Value(
                      bits_mod.UBits(
                          value=int(elem_ast.value),
                          bit_count=elem_type.bit_width)))

            self.global_decls_[name] = CVar(
                ir_value.Value.make_array(values),
                ArrayType(elem_type, len(values)))
          else:
            raise NotImplementedError("Only const arrays supported for now")
        elif isinstance(child.type, c_ast.Enum):
          enum_ast = child.type
          enum_list = enum_ast.values
          assert isinstance(enum_list, c_ast.EnumeratorList)
          enum_curr_val = 0
          width = 32
          is_signed = True
          is_native = True
          for val in enum_list.enumerators:
            assert isinstance(val, c_ast.Enumerator)
            assert val.name not in self.global_decls_
            if val.value is None:
                const_type = IntType(width, is_signed, is_native)
            else:
                const_val, const_type = parse_constant(val.value)
                enum_curr_val = int (const_val) 
            const_expr = ir_value.Value(
                bits_mod.UBits(
                    value= enum_curr_val, bit_count=const_type.bit_width))
            self.global_decls_[val.name] = CVar(const_expr, const_type)
            enum_curr_val += 1
        else:
          raise NotImplementedError("ERROR: Unknown declaration at " +\
                                    str(child.coord))
      else:
        raise NotImplementedError("ERROR: Unknown construct at " + \
                                  str(child.coord))
  


  def parse_type(self, ast_in):
    """Parses a C type's AST.

    Args:
      ast_in: pycparser type AST
    Returns:
      A Type object representing the C type
    """
    is_ref = False
    is_const = False

    ast = ast_in

    if isinstance(ast, c_ast.RefDecl):
      ast = ast.type
      is_ref = True

    is_const = ("const" in ast.quals) if isinstance(ast,
                                                    c_ast.TypeDecl) else False
    array_type = None
    #print("Ast is " + str(ast))
    if isinstance(ast, c_ast.TypeDecl):
      ident = ast.type
      assert isinstance(ident, c_ast.IdentifierType) or isinstance(
          ident, c_ast.TemplateInst)
      if isinstance(ident, c_ast.TemplateInst):
        ident = ident.expr
    elif isinstance(ast, c_ast.IdentifierType):
      ident = ast
    elif isinstance(ast, c_ast.ArrayDecl):
      array_type = self.parse_type(ast.type)
      size, size_type = parse_constant(ast.dim)
      assert isinstance(size_type, IntType)
      ret_type = ArrayType(array_type, size)
      # Arrays are always passed by reference
      ret_type.is_ref = True
      ret_type.is_const = is_const
      return ret_type
    elif isinstance(ast, c_ast.Struct):
      ret_type = StructType(ast.name, ast)
      self.hls_types_by_name_[ret_type.name] = ret_type
      print("HLS types are : " + str(self.hls_types_by_name_))
      return ret_type
    else:
      print(ast)
      raise NotImplementedError("Unimplemented construct ", type(ast))

    assert ident is not None
    is_unsigned = False

    if len(ident.names) == 1:
      name = ident.names[0]
    elif len(ident.names) == 2:
      if ident.names[0] == "unsigned":
        is_unsigned = True
        name = ident.names[1]
      else:
        raise NotImplementedError("Unsupported qualifier for type",
                                  ident.names[0])
    else:
      raise NotImplementedError("Unsupported qualifiers for type", ident.names)

    m_unsigned = re.search("uai([0-9]+)", name)
    m_signed = re.search("sai([0-9]+)", name)

    struct_type = self.get_struct_type(name)

    ptype = None

    if name == "void":
      ptype = VoidType()
    elif name == "int" or name == "long":
      ptype = IntType(32, not is_unsigned, True)
    elif name == "short":
      ptype = IntType(16, not is_unsigned, True)
    elif name == "char":
      ptype = IntType(8, not is_unsigned, True)
    elif name == "bool":
      ptype = BoolType()
    elif m_unsigned is not None:
      ptype = IntType(int(m_unsigned.group(1)), False, False)
    elif m_signed is not None:
      ptype = IntType(int(m_signed.group(1)), True, False)
    elif struct_type is not None:
      assert isinstance(struct_type, hls_types_pb2.HLSStructType)
      ptype = StructType(name, struct_type)
    elif name == "ac_channel":
      assert len(ast.type.params_list.exprs) == 1
      channel_type = self.parse_type(ast.type.params_list.exprs[0])
      ptype = ChannelType(channel_type)
      assert is_ref and not is_const
    else:
      raise NotImplementedError("Unsupported type:", ast_in)

    ptype.is_ref = is_ref
    ptype.is_const = is_const

    return ptype

  def get_global_constant(self, name, loc):
    const_val = self.global_decls_[name].fb_expr
    if name not in self.global_literals:
      self.global_literals[name] = self.fb.add_literal_value(const_val, loc)
    return self.global_literals[name], self.global_decls_[name].ctype

  def gen_default_init(self, decl_type, loc_ast):
    """Generates default RValue for C Type.

    Args:
      decl_type: pycparser AST type
      loc_ast: pycparser AST location
    Returns:
      XLR IR function builder value
    """

    loc = translate_loc(loc_ast)
    assert loc is not None
    if isinstance(decl_type, IntType):
      return self.fb.add_literal_bits(
          bits_mod.UBits(value=0, bit_count=decl_type.bit_width), loc)
    elif isinstance(decl_type, BoolType):
      return self.fb.add_literal_bits(
          bits_mod.UBits(value=0, bit_count=1), loc)
    elif isinstance(decl_type, ArrayType):
      default_elem = self.gen_default_init(decl_type.get_element_type(),
                                           loc_ast)
      return self.fb.add_array([default_elem] * decl_type.get_size(),
                               decl_type.get_element_type().get_xls_type(self.p)
                               ,
                               loc)

    elif isinstance(decl_type, StructType):
      elements = []
      for elem_type in decl_type.get_element_types():
        elements.append(self.gen_default_init(elem_type, loc_ast))
      return self.fb.add_tuple(elements, loc)
    else:
      raise NotImplementedError("Cannot generate default for type ", elem_type)

  def gen_convert_ir(self, in_expr, in_type, to_type, loc_ast):
    """Generates XLS IR value conversion to C Type.

    Args:
      in_expr: XLS function builder value
      in_type: C type of input
      to_type: C Type to convert to
      loc_ast: pycparser AST location
    Returns:
      XLS function builder converted value
    """

    loc = translate_loc(loc_ast)

    if isinstance(to_type, StructType):
      return in_expr
    if isinstance(to_type, ArrayType):
      return in_expr
    if isinstance(to_type, BoolType):
      return self.gen_bool_convert(in_expr, in_type, loc)

    expr_width = in_type.bit_width

    if expr_width == to_type.bit_width:
      return in_expr
    elif expr_width < to_type.bit_width:
      if not isinstance(in_type, BoolType) \
        and in_type.signed \
        and to_type.signed:
        return self.fb.add_signext(in_expr, to_type.bit_width, loc)
      else:
        return self.fb.add_zeroext(in_expr, to_type.bit_width, loc)
    else:
      return self.fb.add_bit_slice(in_expr, 0, to_type.bit_width, loc)

  def gen_bool_convert(self, in_expr, in_type, loc):
    loc = translate_loc(loc)

    assert in_type.bit_width > 0
    if in_type.bit_width == 1:
      return in_expr
    else:
      const0 = self.fb.add_literal_bits(
          bits_mod.UBits(value=0, bit_count=in_type.bit_width), loc)
      return self.fb.add_ne(in_expr, const0, loc)

  def _generic_assign(self, lvalue_expr, rvalue, rvalue_type, condition, loc):
    """Assigns to either a raw C Var or a compound (struct, array) reference.

    Args:
      lvalue_expr: Pycparser expression for LValue
      rvalue: XLS function builder RValue
      rvalue_type: C type of RValue
      condition: Condition of assignment
      loc: XLS function builder source location
    """

    is_struct = isinstance(lvalue_expr, c_ast.StructRef)
    is_array = isinstance(lvalue_expr, c_ast.ArrayRef)
    if isinstance(lvalue_expr, c_ast.ID):
      self.assign(lvalue_expr.name,
                  rvalue,
                  rvalue_type,
                  condition,
                  loc)
    elif is_struct or is_array:
      self.assign_compound(lvalue_expr,
                           rvalue,
                           rvalue_type,
                           condition,
                           loc)
    else:
      raise NotImplementedError("Assigning non-lvalue", str(type(lvalue_expr)))

  def gen_expr_ir(self, stmt_ast, condition):
    """Generates XLS IR value for C expression.

    Args:
      stmt_ast: pycparser C AST for input expression
      condition: XLS IR function builder value for condition of assignment
    Returns:
      XLS IR function builder value for expression
    """

    loc = translate_loc(stmt_ast)
    if isinstance(stmt_ast, c_ast.UnaryOp):
      assert len(stmt_ast.children()) == 1
      operand, operand_type = self.gen_expr_ir(stmt_ast.expr, condition)
      if stmt_ast.op == "-":
        assert isinstance(operand_type, IntType)
        assert operand_type.signed
        return self.fb.add_neg(operand, loc), operand_type
      elif (stmt_ast.op == "++" or
            stmt_ast.op == "--" or
            stmt_ast.op == "p++" or
            stmt_ast.op == "p--"):    # p prefix means post
        assert isinstance(operand_type, IntType)
        synth_literal = c_ast.Constant("int", "1", stmt_ast.coord)
        synth_op = c_ast.BinaryOp(stmt_ast.op[1],
                                  stmt_ast.expr,
                                  synth_literal, stmt_ast.coord)
        r_value, r_type = self.gen_expr_ir(synth_op, condition)
        self._generic_assign(stmt_ast.expr, r_value, r_type, condition, loc)
        if stmt_ast.op[0] == "p":
          return operand, operand_type
        else:
          return r_value, r_type
      elif stmt_ast.op == "~":
        return self.fb.add_not(operand, loc), operand_type
      elif stmt_ast.op == "!":
        const0 = self.fb.add_literal_bits(
            bits_mod.UBits(value=0, bit_count=operand_type.bit_width), loc)
        eq = self.fb.add_eq(operand, const0, loc)
        return eq, BoolType()
      else:
        raise NotImplementedError("Unimplemented unary operator", stmt_ast.op)
    elif isinstance(stmt_ast, c_ast.BinaryOp):
      assert len(stmt_ast.children()) == 2
      left, left_type = self.gen_expr_ir(stmt_ast.left, condition)
      right, right_type = self.gen_expr_ir(stmt_ast.right, condition)
      left_width = left_type.bit_width
      right_width = right_type.bit_width

      left_signed = left_type.signed if isinstance(left_type,
                                                   IntType) else False
      right_signed = right_type.signed if isinstance(right_type,
                                                     IntType) else False

      result_signed = left_signed
      result_native = left_type.native if isinstance(left_type,
                                                     IntType) else True

      add_cmp_fn = None
      if stmt_ast.op == "<":
        add_cmp_fn = self.fb.add_slt if left_signed else self.fb.add_ult
      elif stmt_ast.op == ">":
        add_cmp_fn = self.fb.add_sgt if left_signed else self.fb.add_ugt
      elif stmt_ast.op == "<=":
        add_cmp_fn = self.fb.add_sle if left_signed else self.fb.add_ule
      elif stmt_ast.op == ">=":
        add_cmp_fn = self.fb.add_sge if left_signed else self.fb.add_uge
      elif stmt_ast.op == "==":
        add_cmp_fn = self.fb.add_eq
      elif stmt_ast.op == "!=":
        add_cmp_fn = self.fb.add_ne

      add_logical_fn = None
      if stmt_ast.op == "||":
        add_logical_fn = self.fb.add_or
      elif stmt_ast.op == "&&":
        add_logical_fn = self.fb.add_and

      # Basic arithmetic
      if stmt_ast.op == "+" or stmt_ast.op == "-" or stmt_ast.op == "*":
        binary_op = None
        result_width = max(left_width,
                           right_width) + (0 if result_native else 1)
        if stmt_ast.op == "+":
          binary_op = self.fb.add_add
        elif stmt_ast.op == "-":
          binary_op = self.fb.add_sub
        elif stmt_ast.op == "*":
          result_width = left_width if result_native else (left_width +
                                                           right_width)
          add_mul = self.fb.add_smul if left_signed else self.fb.add_umul
          binary_op = add_mul
        else:
          raise NotImplementedError("Unsupported binary operator", stmt_ast.op)
        if not (isinstance(left_type, IntType) and
                isinstance(right_type, IntType)):
          raise ValueError("Invalid binary operand at " + str(stmt_ast.coord))

        return binary_op(
            self.gen_convert_ir(left, left_type,
                                IntType(result_width,
                                        left_signed,
                                        False),
                                stmt_ast.left),
            self.gen_convert_ir(right, right_type,
                                IntType(result_width,
                                        right_signed,
                                        False),
                                stmt_ast.right),
            loc), IntType(result_width, result_signed, result_native)
      elif stmt_ast.op == "<<":
        result_width = left_width
        result_type = IntType(result_width, left_signed, result_native)
        return self.fb.add_shll(
            self.gen_convert_ir(left, left_type, result_type, stmt_ast.left),
            self.gen_convert_ir(right, right_type, result_type, stmt_ast.right),
            loc), result_type
      elif stmt_ast.op == ">>":
        result_width = left_width
        result_type = IntType(result_width, left_signed, result_native)
        right_converted = self.gen_convert_ir(right,
                                              right_type,
                                              result_type,
                                              stmt_ast.right)
        shr_fn = self.fb.add_shra if left_signed else self.fb.add_shrl
        return shr_fn(
            self.gen_convert_ir(left, left_type, result_type, stmt_ast.left),
            right_converted, loc), result_type
      elif stmt_ast.op == "&":
        result_width = left_width if result_native else max(
            left_width, right_width)
        result_type = IntType(result_width, left_signed, result_native)
        return self.fb.add_and(
            self.gen_convert_ir(left, left_type, result_type, stmt_ast.left),
            self.gen_convert_ir(right, right_type, result_type, stmt_ast.right),
            loc), result_type
      elif stmt_ast.op == "|":
        result_width = left_width if result_native else max(
            left_width, right_width)
        result_type = IntType(result_width, left_signed, result_native)
        return self.fb.add_or(
            self.gen_convert_ir(left, left_type, result_type, stmt_ast.left),
            self.gen_convert_ir(right, right_type, result_type, stmt_ast.right),
            loc), result_type
      elif stmt_ast.op == "^":
        result_width = left_width if result_native else max(
            left_width, right_width)
        result_type = IntType(result_width, left_signed, result_native)
        return self.fb.add_xor(
            self.gen_convert_ir(left, left_type, result_type, stmt_ast.left),
            self.gen_convert_ir(right, right_type, result_type, stmt_ast.right),
            loc), result_type
      elif add_cmp_fn is not None:
        if not (isinstance(left_type, BoolType) or
                isinstance(left_type, IntType)):
          raise ValueError("Invalid operand at " + str(stmt_ast.coord))
        if not (isinstance(right_type, BoolType) or
                isinstance(right_type, IntType)):
          raise ValueError("Invalid operand at " + str(stmt_ast.coord))
        if left_signed != right_signed:
          print("WARNING: Sign mismatch in comparison at " +
                str(stmt_ast.coord))
        result_width = left_width
        left_conv = self.gen_convert_ir(left,
                                        left_type,
                                        IntType(result_width,
                                                left_signed, False),
                                        stmt_ast.left)
        right_conv = self.gen_convert_ir(
            right,
            right_type,
            IntType(result_width, left_signed, False),
            stmt_ast.right)
        return add_cmp_fn(left_conv, right_conv, loc), BoolType()
      elif add_logical_fn is not None:
        left_nz = self.gen_bool_convert(left, left_type, stmt_ast.left)
        right_nz = self.gen_bool_convert(right, right_type, stmt_ast.right)
        return add_logical_fn(left_nz, right_nz, loc), BoolType()
      else:
        raise NotImplementedError("Unknown operator:", stmt_ast.op)
    elif isinstance(stmt_ast, c_ast.TernaryOp):
      cond_expr_raw, cond_type = self.gen_expr_ir(stmt_ast.cond, condition)
      cond_expr = self.gen_bool_convert(cond_expr_raw, cond_type, stmt_ast.cond)
      left_expr, left_type = self.gen_expr_ir(stmt_ast.iftrue, condition)
      right_expr, right_type = self.gen_expr_ir(stmt_ast.iffalse, condition)

      if not isinstance(left_type, type(right_type)):
        raise ValueError("Options for ternary should have the same type")
      if isinstance(left_type, IntType):
        if left_type.signed != right_type.signed:
          raise ValueError("Options for ternary should have the same "
                           "signedness")
        if left_type.bit_width != right_type.bit_width:
          raise ValueError("Options for ternary should have the same width")

      return self.fb.add_sel(cond_expr, left_expr, right_expr, loc), left_type

    elif isinstance(stmt_ast, c_ast.ID):
      if stmt_ast.name in self.cvars:
        if self.cvars[stmt_ast.name] is None:
          raise ValueError("ERROR: Uninitialized variable: ",
                           stmt_ast.name, "at " + str(stmt_ast.coord))
        param = self.cvars[stmt_ast.name].fb_expr
        return param, self.cvars[stmt_ast.name].ctype
      elif stmt_ast.name in self.global_decls_:
        return self.get_global_constant(stmt_ast.name, translate_loc(stmt_ast))
      else:
        raise ValueError("ERROR: Unknown variable", stmt_ast.name,
                         "at " + str(stmt_ast.coord))

    elif isinstance(stmt_ast, c_ast.Constant):
      value, ctype = parse_constant(stmt_ast)
      return self.fb.add_literal_bits(
          bits_mod.UBits(value=value, bit_count=ctype.bit_width),
          translate_loc(stmt_ast)), ctype
    elif isinstance(stmt_ast, c_ast.FuncCall):
      # Check for built in integer types
      if isinstance(stmt_ast.name, c_ast.ID):
        id_name = stmt_ast.name.name
        if id_name in self.functions_:
          func = self.functions_[id_name]
          args_bvalues = []

          assert isinstance(stmt_ast.args, c_ast.ExprList)

          if len(stmt_ast.args.exprs) != len(func.params):
            raise ValueError("Wrong number of args for function call")

          param_types_array = []

          for name_and_type in func.params.items():
            param_types_array.append(name_and_type)

          for arg_idx in range(0, len(stmt_ast.args.exprs)):
            stmt = stmt_ast.args.exprs[arg_idx]
            arg_expr, arg_expr_type = self.gen_expr_ir(stmt, condition)
            _, arg_type = param_types_array[arg_idx]
            conv_arg = self.gen_convert_ir(arg_expr,
                                           arg_expr_type,
                                           arg_type,
                                           stmt)
            args_bvalues.append(conv_arg)

          invoke_returned = self.fb.add_invoke(args_bvalues, func.fb_expr, loc)

          # Handle references
          void_return = isinstance(func.return_type, VoidType)
          unpacked_returns = []
          unpacked_return_types = []

          params_by_index = []

          for name in func.params:
            params_by_index.append(func.params[name])

          if func.count_returns() == 1:
            unpacked_returns.append(invoke_returned)
            if not void_return:
              unpacked_return_types.append(func.return_type)
            else:
              ref_idx = func.get_ref_param_indices()[0]
              unpacked_return_types.append(params_by_index[ref_idx])

          elif func.count_returns() > 1:
            for idx in range(0, func.count_returns()):
              unpacked_returns.append(
                  self.fb.add_tuple_index(invoke_returned, idx, loc))
              if idx == 0 and not void_return:
                unpacked_return_types.append(func.return_type)
              else:
                unpacked_return_types.append(
                    params_by_index[func.get_ref_param_indices()[idx-1]])

          ret_val = None
          if not void_return:
            ret_val = unpacked_returns[0]
            del unpacked_returns[0]
            del unpacked_return_types[0]

          ref_params = func.get_ref_param_indices()
          for ref_idx in range(len(unpacked_returns)):
            param_idx = ref_params[ref_idx]
            expr = stmt_ast.args.exprs[param_idx]

            self._generic_assign(expr,
                                 unpacked_returns[ref_idx],
                                 unpacked_return_types[ref_idx],
                                 condition,
                                 translate_loc(stmt_ast))

          return ret_val, func.return_type
        else:
          raise ValueError("ERROR: Unknown function", id_name)
      elif isinstance(stmt_ast.name, c_ast.StructRef):
        struct_ast = stmt_ast.name
        left_var = struct_ast.name
        assert struct_ast.type == "."
        template_ast = struct_ast.field
        # TODO(seanhaskell): What's going on here? Strange C AST form
        if isinstance(template_ast, c_ast.ID):
          template_ast = template_ast.name
        if isinstance(template_ast, c_ast.TemplateInst):
          if template_ast.expr == "slc":
            left_fb, left_type = self.gen_expr_ir(left_var, condition)
            assert len(template_ast.params_list.exprs) == 1
            width_expr = template_ast.params_list.exprs[0]
            # TODO(seanhaskell): Allow const expressions to specify width
            assert isinstance(width_expr, c_ast.Constant)
            width = int(width_expr.value)
            assert isinstance(stmt_ast.args, c_ast.ExprList)
            assert len(stmt_ast.args.exprs) == 1
            offset_expr = stmt_ast.args.exprs[0]
            if isinstance(offset_expr, c_ast.Constant):
              ret_fb = self.fb.add_bit_slice(left_fb,
                                             int(offset_expr.value),
                                             width,
                                             loc)
            else:
              offset_val, offset_type = self.gen_expr_ir(offset_expr, condition)
              assert isinstance(offset_type, IntType)
              shift_fb = self.fb.add_shrl(left_fb, offset_val)
              ret_fb = self.fb.add_bit_slice(shift_fb,
                                             0,   # Already shifted
                                             width,
                                             loc)

            return ret_fb, IntType(width, left_type.signed, False)
          else:
            raise NotImplementedError("Unknown non-template function on int",
                                      template_ast.expr)
        elif template_ast == "set_slc":
          assert len(stmt_ast.args.exprs) == 2
          offset_ast = stmt_ast.args.exprs[0]
          assert isinstance(offset_ast, c_ast.Constant)
          offset, offset_type = parse_constant(offset_ast)
          assert isinstance(offset_type, IntType)
          value_expr, value_type = self.gen_expr_ir(stmt_ast.args.exprs[1],
                                                    condition)

          # Check rvalue type
          assert isinstance(value_type, IntType)
          assert not value_type.native

          l_o_value, l_o_type = self.gen_expr_ir(left_var, condition)
          assert isinstance(l_o_type, IntType)

          concat_list = [value_expr]

          if offset > 0:
            concat_list.append(
                self.fb.add_bit_slice(l_o_value, 0, offset, loc))

          right_hand_bits = l_o_type.bit_width - (
              offset + value_type.bit_width)
          if right_hand_bits > 0:
            concat_list.insert(
                0,
                self.fb.add_bit_slice(l_o_value,
                                      offset + value_type.bit_width,
                                      right_hand_bits, loc))

          r_expr = self.fb.add_concat(concat_list, loc)
          r_type = IntType(l_o_type.bit_width, l_o_type.signed, False)

          self._generic_assign(left_var, r_expr, r_type, condition,
                               translate_loc(stmt_ast))
        elif template_ast == "read":
          assert isinstance(left_var, c_ast.ID)
          assert stmt_ast.args is None
          left_fb, left_type = self.gen_expr_ir(left_var, condition)
          assert isinstance(left_type, ChannelType)
          assert left_type.is_input
          self.read_tokens[id(left_fb)] = left_var.name
          return left_fb, left_type.channel_type
        elif template_ast == "write":
          assert len(stmt_ast.args.exprs) == 1
          obj = stmt_ast.args.exprs[0]
          assert isinstance(left_var, c_ast.ID)

          obj_expr, obj_type = self.gen_expr_ir(obj, condition)

          in_ch_name = self.read_tokens[id(obj_expr)]
          out_ch_name = left_var.name

          # z
          self.assign(out_ch_name, obj_expr, obj_type, condition, loc)
          # in vz -> out lz
          self.assign(out_ch_name + "_lz",
                      self.cvars[in_ch_name + "_vz"].fb_expr,
                      self.cvars[in_ch_name + "_vz"].ctype, condition, loc)
          # out vz -> in lz
          self.assign(in_ch_name + "_lz",
                      self.cvars[out_ch_name + "_vz"].fb_expr,
                      self.cvars[out_ch_name + "_vz"].ctype, condition, loc)

          # vz_name = right_name + "_vz"
          # lz_name = right_name + "_lz"

          return None, VoidType()
        else:
          raise NotImplementedError("Unsupported method", template_ast)
      else:
        raise NotImplementedError("Unsupported construct", type(stmt_ast.name))
    elif isinstance(stmt_ast, c_ast.Cast):
      parsed_type = self.parse_type(stmt_ast.to_type)
      if isinstance(parsed_type, IntType) or isinstance(parsed_type, BoolType):
        fb_expr, fb_type = self.gen_expr_ir(stmt_ast.expr, condition)
        return self.gen_convert_ir(fb_expr,
                                   fb_type,
                                   parsed_type,
                                   stmt_ast.expr), parsed_type
      else:
        raise NotImplementedError("Only integer casts supported")
    elif isinstance(stmt_ast, c_ast.ArrayRef):
      index_expr, index_type = self.gen_expr_ir(stmt_ast.subscript, condition)
      assert isinstance(index_type, IntType)

      left_expr, left_type = self.gen_expr_ir(stmt_ast.name, condition)
      if left_expr is not None:
        if isinstance(left_type, ArrayType):
          return self.fb.add_array_index(left_expr, index_expr,
                                         loc), left_type.get_element_type()
        elif isinstance(left_type, IntType):
          # TODO(seanhaskell): Allow variable offset.
          # TODO(seanhaskell): Need function_builder support
          offset_expr = stmt_ast.subscript
          assert isinstance(offset_expr, c_ast.Constant)
          return self.fb.add_bit_slice(left_expr, int(offset_expr.value), 1,
                                       loc), IntType(1, False, False)
        else:
          raise NotImplementedError("Array reference of unsupported "
                                    "lparam type:", type(left_type))

      else:
        assert isinstance(stmt_ast.name, c_ast.ID)
        assert stmt_ast.name.name in self.global_decls_
        # Re-use literals
        array_literal, element_type = self.get_global_constant(
            stmt_ast.name.name, translate_loc(stmt_ast.name))
        return self.fb.add_array_index(array_literal, index_expr,
                                       loc), element_type
    elif isinstance(stmt_ast, c_ast.StructRef):
      left_expr, left_type = self.gen_expr_ir(stmt_ast.name, condition)

      assert isinstance(left_type, StructType)
      assert isinstance(stmt_ast.field, c_ast.ID)
      right_name = stmt_ast.field.name

      field_indices = left_type.get_field_indices()
      assert right_name in field_indices
      field_index = field_indices[right_name]

      return self.fb.add_tuple_index(
          left_expr, field_index, loc), left_type.get_element_type(right_name)
    else:
      raise NotImplementedError("ERROR: Unsupported expression AST: ",
                                type(stmt_ast))

  def gen_ir(self):
    """Generates XLS IR value for C source file, once parse()'d.

    Returns:
      A string containing the resulting XLS IR
    """

    p = ir_package.Package(self.package_name_)

    for f_name, func in self.functions_.items():

      self.fb = function_builder.FunctionBuilder(f_name, p)
      self.p = p

      # Parameter / variable IR values by name
      # Parameter / variable types by name
      self.cvars = {}
      # Variable redirection: const is False, normal is True,
      #     references have name
      self.lvalues = {}
      # Literals for const arrays
      self.global_literals = {}
      # For break/continue
      self.continue_condition = None
      self.break_condition = None
      # Input read tokens
      self.read_tokens = {}

      self.channel_params = []

      for p_name, ptype in func.params.items():
        if not isinstance(ptype, ChannelType):
          self.cvars[p_name] = CVar(
              self.fb.add_param(p_name, ptype.get_xls_type(p), func.loc), ptype)
          self.lvalues[p_name] = not ptype.is_const
        else:
          bit_type = BoolType()
          self.cvars[p_name + "_vz"] = CVar(
              self.fb.add_param(p_name + "_vz", p.get_bits_type(1), func.loc),
              bit_type)
          self.channel_params.append((p_name + "_vz", 1))
          self.cvars[p_name + "_lz"] = CVar(
              self.gen_default_init(bit_type, func.loc), bit_type)
          self.lvalues[p_name + "_lz"] = True
          if ptype.is_input:
            self.cvars[p_name] = CVar(
                self.fb.add_param(p_name + "_z",
                                  ptype.channel_type.get_xls_type(p), func.loc),
                ptype)
            self.channel_params.append(
                (p_name + "_z", ptype.channel_type.bit_width))
          else:
            self.cvars[p_name] = CVar(
                self.gen_default_init(ptype.channel_type, func.loc),
                ptype.channel_type)
            self.lvalues[p_name] = True

      # Function body
      ret_vals = self.gen_ir_block(func.body_ast.children(), None, None)

      # Add in reference params
      reference_param_names = func.get_ref_param_names()

      returns = []

      # Combine returns
      if not isinstance(func.return_type, VoidType):
        # Last return should be unconditional
        default_return_index = -1
        for idx in range(0, len(ret_vals)):
          if ret_vals[idx].condition is None:
            assert default_return_index < 0
            default_return_index = idx

        if default_return_index < 0:
          print("WARNING: Adding 0 for default (unqualified) return")
          ret_vals.append(
              RetVal(
                  None,
                  self.fb.add_literal_value(
                      ir_value.Value(
                          bits_mod.UBits(
                              value=0, bit_count=func.return_type.bit_width)),
                      func.loc), func.return_type))
          default_return_index = len(ret_vals) - 1

        assert default_return_index >= 0
        ret_expr = self.gen_convert_ir(ret_vals[default_return_index].fb_expr,
                                       ret_vals[default_return_index]\
                                       .return_type,
                                       func.return_type,
                                       func.loc)

        for i in reversed(range(0, len(ret_vals))):
          if i == default_return_index:
            continue

          conv_expr = self.gen_convert_ir(ret_vals[i].fb_expr,
                                          ret_vals[i].return_type,
                                          func.return_type,
                                          func.loc)
          ret_expr = self.fb.add_sel(ret_vals[i].condition, conv_expr, ret_expr,
                                     func.loc)

        assert ret_expr is not None
        returns.append(ret_expr)

      if reference_param_names:
        returns = returns + list(
            [self.cvars[name].fb_expr for name in reference_param_names])

      # Add channel returns
      self.channel_returns = []
      for pname, ptype in func.params.items():
        if isinstance(ptype, ChannelType):
          returns.append(self.cvars[pname + "_lz"].fb_expr)
          self.channel_returns.append((pname + "_lz", 1))
          if not ptype.is_input:
            returns.append(self.cvars[pname].fb_expr)
            self.channel_returns.append(
                (pname + "_z", ptype.channel_type.bit_width))

      if func.count_returns() > 1:
        self.fb.add_tuple(returns, func.loc)
      elif func.count_returns() == 1:
        # Ensure return value is the last thing
        self.fb.add_identity(returns[0], func.loc)
      else:
        raise NotImplementedError("ERROR: Void function with no reference"
                                  " parameters doesn't make sense with XLS")

      func.set_fb_expr(self.fb.build())

    return p

  def gen_ir_block_compound_or_single(self,
                                      ast,
                                      condition,
                                      inject_local_vars,
                                      switch_case_block=False):
    """Generate XLS IR for either a single ASR expression or a compound.

    Args:
      ast: C AST root
      condition: Condition for assignments
      inject_local_vars: Extra local variables, like counter in for loop
      switch_case_block: Inside case?
    Returns:
      List of RetVals
    """

    true_stmts = []
    if isinstance(ast, c_ast.Compound):
      true_stmts += ast.block_items if (ast.block_items is not None) else []
    elif isinstance(ast, list):
      true_stmts = ast
    else:
      true_stmts = [ast]

    return self.gen_ir_block(
        [("dummy", x) for x in true_stmts],
        condition, inject_local_vars, switch_case_block)

  def assign(self, name, r_expr, r_type, condition, loc):
    """Generate an assignment (LValue).

    Args:
      name: Variable name
      r_expr: XLS function builder RValue
      r_type: C type of RValue
      condition: Condition of assignment
      loc: XLS function builder source location
    """

    if name not in self.cvars:
      raise ValueError("ERROR: Assignment to unknown variable", name)
    # Check constness
    if not self.lvalues[name]:
      raise NotImplementedError("Asignment to const variable ", name)
    assert self.lvalues[name]

    loc = translate_loc(loc)
    converted = self.gen_convert_ir(r_expr,
                                    r_type,
                                    self.cvars[name].ctype,
                                    loc)

    if self.continue_condition is not None:
      condition = self.combine_condition_can_be_none(
          condition, self.fb.add_not(self.continue_condition, loc), True, loc)
    if self.break_condition is not None:
      condition = self.combine_condition_can_be_none(
          condition, self.fb.add_not(self.break_condition, loc), True, loc)
    if condition is not None:
      converted = self.fb.add_sel(condition, converted,
                                  self.cvars[name].fb_expr, loc)
    self.cvars[name].fb_expr = converted

  def assign_compound(self, lvalue_ast, r_expr, r_type, condition, loc):
    """Generate an assignment to a compound type (Struct, Array).

    Args:
      lvalue_ast: Variable C AST
      r_expr: XLS function builder RValue
      r_type: C type of RValue
      condition: Condition of assignment
      loc: XLS function builder source location
    """

    compound_node = lvalue_ast
    compound_lvals = [compound_node]
    while isinstance(compound_node.name, c_ast.StructRef) or \
        isinstance(compound_node.name, c_ast.ArrayRef):
      compound_node = compound_node.name
      compound_lvals = compound_lvals + [compound_node]

    assert isinstance(compound_lvals[-1].name, c_ast.ID)
    assign_name = compound_lvals[-1].name.name
    if assign_name not in self.cvars:
      raise ValueError("ERROR: Assignment to unknown variable", assign_name)

    for compound_lval in compound_lvals:
      left_expr, left_type = self.gen_expr_ir(compound_lval.name, condition)

      # Build up expression
      if isinstance(compound_lval, c_ast.StructRef):

        element_values = []

        field_indices = left_type.get_field_indices()
        field_index = field_indices[compound_lval.field.name]
        element_type = left_type.get_element_type(compound_lval.field.name)
        r_expr = self.gen_convert_ir(r_expr, r_type, element_type, lvalue_ast)
        for _, index in field_indices.items():
          if index == field_index:
            element_values.append(r_expr)
          else:
            element_values.append(
                self.fb.add_tuple_index(left_expr, index, loc))

        r_expr = self.fb.add_tuple(element_values, loc)
        r_type = left_type
      elif isinstance(compound_lval, c_ast.ArrayRef):
        if not isinstance(compound_lval.subscript, c_ast.Constant):
          raise NotImplementedError("Variable array indexing")
        elem_index, elem_type = parse_constant(compound_lval.subscript)
        if not isinstance(elem_type, IntType):
          raise ValueError("Array index must be integer")

        element_values = []

        r_expr = self.gen_convert_ir(r_expr,
                                     r_type,
                                     left_type.get_element_type(),
                                     lvalue_ast)

        for index in range(0, left_type.get_size()):
          if index == elem_index:
            element_values.append(r_expr)
          else:
            element_values.append(
                self.fb.add_array_index(
                    left_expr,
                    self.fb.add_literal_bits(
                        bits_mod.UBits(value=index,
                                       bit_count=elem_type.bit_width),
                        loc), loc))

        r_expr = self.fb.add_array(
            element_values,
            left_type.get_element_type().get_xls_type(self.p), loc)
        r_type = left_type
      else:
        raise NotImplementedError("Unknown compound in assignment",
                                  type(compound_lval))

    self.assign(assign_name, r_expr, r_type, condition, loc)

  def combine_condition_can_be_none(self, acond, bcond, use_and, loc):
    if (acond is not None) and (bcond is not None):
      return self.fb.add_and(acond, bcond, loc) if use_and else self.fb.add_or(
          acond, bcond, loc)
    else:
      if acond is not None:
        return acond
      else:
        return bcond

  def gen_ir_block(self,
                   stmt_list,
                   condition,
                   inject_local_vars,
                   switch_case_block=False):
    """Generate an XLS IR block from a C AST block.

    Args:
      stmt_list: List of pycparser C AST nodes
      condition: Condition of assignments
      inject_local_vars: Extra local variables, like in a for loop
      switch_case_block: Inside of a switch case?
    Returns:
      List of RetVal
    """

    # Private context (variables etc)
    prev_cvars = self.cvars
    self.cvars = copy.copy(self.cvars)

    if inject_local_vars is not None:
      for name, cvar in inject_local_vars.items():
        if name in self.cvars:
          raise ValueError("Variable '", name,
                           "' already declared in this scope")
        self.cvars[name] = cvar

    ret_vals = []

    next_line_pragma = None
    for _, stmt in stmt_list:
      loc = translate_loc(stmt)

      this_line_pragma = next_line_pragma
      next_line_pragma = None

      if isinstance(stmt, c_ast.Return):
        ret_val, ret_type = self.gen_expr_ir(stmt.expr, condition)
        ret_vals.append(RetVal(condition, ret_val, ret_type))
        break  # Ignore the rest of the block
      elif isinstance(stmt, c_ast.Decl):
        if stmt.name in self.cvars:
          raise ValueError("Variable '", stmt.name,
                           "' already declared in this scope")

        decl_type = self.parse_type(stmt.type)
        self.cvars[stmt.name] = CVar(None, decl_type) 
        self.lvalues[stmt.name] = not decl_type.is_const
        if stmt.init is not None:
          if isinstance(stmt.init, c_ast.InitList):
            if not isinstance(decl_type, ArrayType):
              raise NotImplementedError("Initializer list for non-array")
            if len(stmt.init.exprs) != decl_type.get_size():
              raise ValueError("Number of initializers doesn't match"
                               " array size")
            elements = []
            for index in range(0, decl_type.get_size()):
              init_expr_ast = stmt.init.exprs[index]
              init_expr, init_type = self.gen_expr_ir(init_expr_ast, condition)
              elements.append(
                  self.gen_convert_ir(init_expr,
                                      init_type,
                                      decl_type.get_element_type(),
                                      init_expr_ast))
            self.cvars[stmt.name].fb_expr = self.fb.add_array(
                elements,
                decl_type.get_element_type().get_xls_type(self.p), loc)
          else:
            init_val, init_type = self.gen_expr_ir(stmt.init, condition)
            self.cvars[stmt.name].fb_expr = self.gen_convert_ir(
                init_val, init_type, decl_type, stmt)
        else:
          self.cvars[stmt.name].fb_expr = self.gen_default_init(decl_type, stmt)
      elif isinstance(stmt, c_ast.Assignment) or isinstance(
          stmt, c_ast.FuncCall):
        if isinstance(stmt, c_ast.FuncCall):
          self.gen_expr_ir(stmt, condition)
        else:
          synth_op = None

          if stmt.op == "=":
            synth_op = stmt.rvalue
          else:
            assert stmt.op[len(stmt.op) - 1] == "="
            synth_op = c_ast.BinaryOp(stmt.op[0:len(stmt.op) - 1], stmt.lvalue,
                                      stmt.rvalue, stmt.coord)

          r_expr, r_type = self.gen_expr_ir(synth_op, condition)

          self._generic_assign(stmt.lvalue, r_expr, r_type, condition, loc)
      elif isinstance(stmt, c_ast.Pragma):
        next_line_pragma = stmt
      elif isinstance(stmt, c_ast.For):
        if (self.continue_condition is not None) or \
           (self.break_condition is not None):
          raise NotImplementedError("ERROR: TODO: Nested for loops")

        self.continue_condition = None
        self.break_condition = None

        unroll = False
        if this_line_pragma is not None:
          if this_line_pragma.string == "hls_unroll yes":
            unroll = True
          else:
            raise ValueError("Unknown pragma:", this_line_pragma.string)

        if not unroll:
          raise NotImplementedError(
              "Only unrolled loops supported for now. "
              "Use #pragma hls_unroll yes")

        # Enforce a simple form
        assert isinstance(stmt.init, c_ast.DeclList)
        assert len(stmt.init.decls) == 1

        decl = stmt.init.decls[0]
        assert isinstance(decl, c_ast.Decl)
        counter_name = decl.name
        assert counter_name not in self.cvars
        counter_type = self.parse_type(decl.type)
        assert isinstance(decl.init, c_ast.Constant)
        counter_start = int(decl.init.value)
        assert counter_type.bit_width == 32
        assert counter_type.native
        assert counter_type.signed

        assert isinstance(stmt.cond, c_ast.BinaryOp)
        assert stmt.cond.op == "<"
        assert isinstance(stmt.cond.left, c_ast.ID)
        assert stmt.cond.left.name == counter_name
        assert isinstance(stmt.cond.right, c_ast.Constant)
        counter_end = int(stmt.cond.right.value)

        for count in range(counter_start, counter_end):

          counter_cvar = CVar(
              self.fb.add_literal_bits(
                  bits_mod.UBits(value=count, bit_count=32), loc),
              IntType(32, True, True))

          c_ast.Constant(decl.type, count)

          self.continue_condition = None
          ret_vals += self.gen_ir_block_compound_or_single(
              stmt.stmt, condition, {counter_name: counter_cvar})

      elif isinstance(stmt, c_ast.If):
        cond_expr_raw, cond_type = self.gen_expr_ir(stmt.cond, condition)
        cond_expr = self.gen_bool_convert(cond_expr_raw, cond_type, stmt.cond)

        if condition is None:
          compound_condition = cond_expr
        else:
          compound_condition = self.fb.add_and(condition, cond_expr, loc)

        ret_vals += self.gen_ir_block_compound_or_single(stmt.iftrue,
                                                         compound_condition,
                                                         None)

        if stmt.iffalse is not None:
          not_cond_expr = self.fb.add_not(cond_expr, loc)

          if condition is None:
            not_compound_condition = not_cond_expr
          else:
            not_compound_condition = self.fb.add_and(condition, not_cond_expr,
                                                     loc)

          ret_vals += \
              self.gen_ir_block_compound_or_single(stmt.iffalse,
                                                   not_compound_condition,
                                                   None)

      elif isinstance(stmt, c_ast.Continue) or isinstance(stmt, c_ast.Break):
        if not switch_case_block:
          if isinstance(stmt, c_ast.Continue):
            self.continue_condition = self.combine_condition_can_be_none(
                self.continue_condition, condition, False, translate_loc(stmt))
            break  # Ignore the rest of the block
          else:
            self.break_condition = self.combine_condition_can_be_none(
                self.break_condition, condition, False, translate_loc(stmt))
            break  # Ignore the rest of the block
        else:
          if isinstance(stmt, c_ast.Break):
            break  # Ignore the rest of the block
          else:
            raise NotImplementedError("continue in switch")
      elif isinstance(stmt, c_ast.Switch):
        cond_expr, cond_type = self.gen_expr_ir(stmt.cond, condition)
        if not isinstance(cond_type, IntType):
          raise ValueError("Switch must be on integer")
        assert isinstance(stmt.stmt, c_ast.Compound)
        next_is_default = False
        case_falls_thru = False
        fall_thru_cond = None
        for item in stmt.stmt.block_items:
          if isinstance(item, c_ast.Case):
            case_expr, case_type = self.gen_expr_ir(item.expr, condition)
            conv_case = self.gen_convert_ir(
                case_expr,
                case_type,
                IntType(cond_type.bit_width, cond_type.signed, False),
                item)
            case_loc = translate_loc(item.expr)
            case_condition = self.fb.add_eq(cond_expr, conv_case, case_loc)
            
            if case_falls_thru:
                case_condition = self.fb.add_or(fall_thru_cond, case_condition, case_loc)
                case_falls_thru = False
            else:
                fall_thru_cond = case_condition                         
            if condition is None:
              compound_condition = case_condition
            else:
              compound_condition = self.fb.add_and(condition, case_condition,
                                                   case_loc)
            ret_stmt = self.gen_ir_block_compound_or_single(
                item.stmts, compound_condition, None, True)
            if not ret_stmt:
              case_falls_thru = True
            if next_is_default:
              ret_vals += self.gen_ir_block_compound_or_single(
                  item.stmts, condition, None, True)
            ret_vals += ret_stmt
            next_is_default = False
          else:
            assert isinstance(item, c_ast.Default)
            ret_stmt = self.gen_ir_block_compound_or_single(
                item.stmts, condition, None, True)
            # TODO(seanhaskell): Also break
            if not ret_stmt:
              next_is_default = True
            else:
              case_falls_thru = False
            ret_vals += ret_stmt
      elif isinstance(stmt, c_ast.EmptyStatement):
        pass
      elif isinstance(stmt, c_ast.UnaryOp):
        self.gen_expr_ir(stmt, condition)
      else:
        stmt.show()
        raise NotImplementedError("Unsupported construct in function body " +
                                  str(type(stmt)))

    # Restore context (variables etc)
    updated_cvars = self.cvars
    self.cvars = prev_cvars

    # Merge contexts (update assignments)
    for name in updated_cvars.keys():
      was_injected = False
      if inject_local_vars is not None:
        was_injected = name in inject_local_vars.keys()

      if (name in self.cvars.keys()) and not was_injected:
        self.cvars[name] = updated_cvars[name]

    return ret_vals
