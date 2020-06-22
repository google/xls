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

# Lint as: python3
"""Specification for XLS ops.

The contents of this file is used to generate op.h and op.cc.
"""

import collections
import enum
from typing import List, Optional, Text


class ConstructorArgument(object):
  """Describes an argument to the Node class constructor.

  Attributes:
    name: Name of the argument.
    cpp_type: The C++ type of the argument.
    clone_expression: The expression to use when calling the constructor when
      cloning.
  """

  def __init__(self,
               name: Text,
               cpp_type: Text,
               clone_expression: Optional[Text] = None):
    self.name = name
    self.cpp_type = cpp_type
    self.clone_expression = clone_expression


class DataMember(object):
  """Describes a data member of the Node class.

  Attributes:
    name: Name of the data member. Should have a trailing '_' to follow the C++
      style guide.
    cpp_type: The C++ type of the data member.
    init: The expression to initialize the data member with. This is evaluated
      in the constructor member initializer list.
    equals_tmpl: A Python format string defining the expression for testing this
      member for equality. The format fields are named 'lhs' and 'rhs'. Example:
        '{lhs}.EqualTo({rhs})'.
  """

  def __init__(self,
               name: Text,
               cpp_type: Text,
               init: Text,
               equals_tmpl: Text = '{lhs} == {rhs}'):
    self.name = name
    self.cpp_type = cpp_type
    self.init = init
    self.equals_tmpl = equals_tmpl


class Method(object):
  """Describes a method of the Node class.

  Attributes:
    name: Name of the method.
    return_cpp_type: The C++ type of the value returned by the method.
    expression: The expression to produce the value returned by the method.
    params: Optional string of C++ parameters that gets inserted into the method
      signature between "()"s.
  """

  def __init__(self,
               name: Text,
               return_cpp_type: Text,
               expression: Optional[Text],
               params: Text = ''):
    self.name = name
    self.return_cpp_type = return_cpp_type
    self.expression = expression
    self.params = params


class Attribute(object):
  """Describes an attribute of a Node class.

  An Attribute desugars into a ConstructorArgument, DataMember, and an accessor
  Method.

  Attributes:
    name: The name of the attribute. The constructor argument and accessor
      method share this name. The data member is the same name with a '_'
      suffix.
    constructor_argument: The ConstructorArgument of the attribute.
    data_member: The DataMember of the attribute.
    method: The accessor Method of the attribute.
  """

  def __init__(self,
               name: Text,
               cpp_type: Text,
               arg_cpp_type: Optional[Text] = None,
               return_cpp_type: Optional[Text] = None,
               equals_tmpl: Text = '{lhs} == {rhs}'):
    """Initialize an Attribute.

    Args:
      name: The name of the attribute. The constructor argument and accessor
        method share this name. The data member is the same name with a '_'
        suffix.
      cpp_type: The C++ type of the data member holding the attribute.
      arg_cpp_type: The C++ type of the constructor argument for passing in the
        attribute value.  Defaults to cpp_type.
      return_cpp_type: The return type of the accessor method for the attribute.
        Defaults to cpp_type.
      equals_tmpl: A Python format string defining the expression for testing
        this member for equality. The format fields are named 'lhs' and 'rhs'.
        For example, '{lhs}.EqualTo({rhs})'.
    """

    self.name = name
    self.constructor_argument = ConstructorArgument(
        name=self.name,
        cpp_type=cpp_type if arg_cpp_type is None else arg_cpp_type,
        clone_expression=self.name + '()')
    self.data_member = DataMember(
        name=name + '_', cpp_type=cpp_type, init=name, equals_tmpl=equals_tmpl)
    self.method = Method(
        name=name,
        return_cpp_type=cpp_type
        if return_cpp_type is None else return_cpp_type,
        expression=self.data_member.name)


class Int64Attribute(Attribute):

  def __init__(self, name):
    super(Int64Attribute, self).__init__(name, cpp_type='int64')


class TypeAttribute(Attribute):

  def __init__(self, name):
    super(TypeAttribute, self).__init__(name, cpp_type='Type*')


class FunctionAttribute(Attribute):

  def __init__(self, name):
    super(FunctionAttribute, self).__init__(
        name,
        cpp_type='Function*',
        equals_tmpl='{lhs}->IsDefinitelyEqualTo({rhs})')


class ValueAttribute(Attribute):

  def __init__(self, name):
    super(ValueAttribute, self).__init__(
        name, cpp_type='Value', return_cpp_type='const Value&')


class StringAttribute(Attribute):

  def __init__(self, name):
    super(StringAttribute, self).__init__(
        name,
        cpp_type='std::string',
        return_cpp_type='const std::string&',
        arg_cpp_type='absl::string_view')


class LsbOrMsbAttribute(Attribute):

  def __init__(self, name):
    super(LsbOrMsbAttribute, self).__init__(
        name,
        cpp_type='LsbOrMsb',
        return_cpp_type='LsbOrMsb',
        arg_cpp_type='LsbOrMsb')


class Property(enum.Enum):
  """Enumeration of properties of Ops.

  An Op can have zero or more properties.
  """
  BITWISE = 1  # Ops such as kXor, kAnd, and kOr
  ASSOCIATIVE = 2
  COMMUTATIVE = 3
  COMPARISON = 4


class Operand(object):

  def __init__(self, name: Text):
    self.name = name
    self.add_method = 'AddOperand'


class OperandSpan(Operand):

  def __init__(self, name: Text):
    super(OperandSpan, self).__init__(name)
    self.name = name
    self.add_method = 'AddOperands'


class OptionalOperand(Operand):

  def __init__(self, name: Text):
    super(OptionalOperand, self).__init__(name)
    self.name = name
    self.add_method = 'AddOptionalOperand'


class OpClass(object):
  """Describes a C++ subclass of xls::Node."""

  # Collection of all OpClass instances.
  kinds = collections.OrderedDict()

  def __init__(self,
               name: Text,
               op: Text,
               operands: List[Operand],
               xls_type_expression: Text,
               attributes: List[Attribute] = (),
               extra_constructor_args=(),
               extra_data_members=(),
               extra_methods: List[Method] = (),
               custom_clone_method: bool = False):
    """Initializes an OpClass.

    Args:
      name: The name of the class.
      op: The expression for the Op associated with this class (e.g.,
        'Op::kParam').
      operands: The list of operands.
      xls_type_expression: The expression evaluated in the constructor member
        initialization list which produces the xls::Type* of this node.
      attributes: List of Attributes of this class.
      extra_constructor_args: List of additional constructor arguments.
      extra_data_members: List of additional data members.
      extra_methods: List of additional class methods.
      custom_clone_method: Whether this class has a custom clone method. If true
        the method should be defined directly in nodes_source.tmpl.
    """
    self.name = name
    self.op = op
    self.operands = operands
    self.xls_type_expression = xls_type_expression
    self.attributes = attributes
    self.extra_constructor_args = extra_constructor_args
    self.extra_data_members = extra_data_members
    self.extra_methods = extra_methods
    self.custom_clone_method = custom_clone_method

  def constructor_args_str(self) -> Text:
    """The constructor arguments list as a single string."""
    args = [
        ConstructorArgument('loc', 'absl::optional<SourceLocation>', 'loc()')
    ]
    for o in self.operands:
      if isinstance(o, OperandSpan):
        args.append(ConstructorArgument(o.name, 'absl::Span<Node* const>'))
      elif isinstance(o, OptionalOperand):
        args.append(ConstructorArgument(o.name, 'absl::optional<Node*>'))
      else:
        args.append(ConstructorArgument(o.name, 'Node*'))
    args.extend(a.constructor_argument for a in self.attributes)
    args.extend(self.extra_constructor_args)
    args.append(ConstructorArgument('function', 'Function*', 'function()'))
    return ', '.join(a.cpp_type + ' ' + a.name for a in args)

  def base_constructor_invocation(self):
    return 'Node({op}, {type_expr}, loc, function)'.format(
        op=self.op, type_expr=self.xls_type_expression)

  def methods(self) -> List[Method]:
    """Returns the methods defined for this class."""
    methods = [a.method for a in self.attributes]
    methods.extend(self.extra_methods)
    return methods

  def clone_args_str(self, new_operands: Text) -> Text:
    """Returns the arguments to pass to the constructor during cloning.

    Args:
      new_operands: The name of the span variable containing the new operands
        during cloning.
    """
    assert not self.custom_clone_method
    args = ['loc()']
    if len(self.operands) == 1 and isinstance(self.operands[0], OperandSpan):
      args.append(new_operands)
    else:
      for i, o in enumerate(self.operands):
        assert isinstance(o, Operand)
        args.append('{}[{}]'.format(new_operands, i))
    args.extend('{}()'.format(a.name) for a in self.attributes)
    args.extend(a.clone_expression for a in self.extra_constructor_args)
    return ', '.join(args)

  def data_members(self) -> List[DataMember]:
    """Returns the data members of the class."""
    members = [a.data_member for a in self.attributes]
    members.extend(self.extra_data_members)
    return members

  def equal_to_expr(self) -> Text:
    """Returns expression used in IsDefinitelyEqualTo to compare expression."""

    def data_member_equal(m):
      lhs = m.name
      rhs = 'other->As<{cls}>()->{name}'.format(cls=self.name, name=m.name)
      return m.equals_tmpl.format(lhs=lhs, rhs=rhs)

    assert self.data_members()
    return '&& '.join(data_member_equal(m) for m in self.data_members())


class Op(object):
  """Describes an xls::Op.

  Attributes:
   enum_name: The name of the C++ enum value (e.g., 'kParam').
   name: The name of the op as it appears in textual IR (e.g., 'param').
   op_class: The OpClass value indicating the C++ Node subclass of the op.
   properties: A List of Properties describing the op.
  """

  def __init__(self, enum_name: Text, name: Text, op_class: OpClass,
               properties: List[Property]):
    self.enum_name = enum_name
    self.name = name
    self.op_class = op_class
    self.properties = properties

# pyformat: disable
OpClass.kinds['ARRAY'] = OpClass(
    name='Array',
    op='Op::kArray',
    operands=[OperandSpan('elements')],
    xls_type_expression='function->package()->GetArrayType(elements.size(), element_type)',
    attributes=[TypeAttribute('element_type')],
    extra_methods=[Method(name='size',
                          return_cpp_type='int64',
                          expression='operand_count()')]
)

OpClass.kinds['ARRAY_INDEX'] = OpClass(
    name='ArrayIndex',
    op='Op::kArrayIndex',
    operands=[Operand('arg'), Operand('index')],
    xls_type_expression='arg->GetType()->AsArrayOrDie()->element_type()',
)

OpClass.kinds['ARRAY_UPDATE'] = OpClass(
    name='ArrayUpdate',
    op='Op::kArrayUpdate',
    operands=[Operand('arg'), Operand('index'), Operand('update_value')],
    xls_type_expression='arg->GetType()',
    extra_methods=[Method(name='size',
                          return_cpp_type='int64',
                          expression='GetType()->AsArrayOrDie()->size()')]
)

OpClass.kinds['BIN_OP'] = OpClass(
    name='BinOp',
    op='op',
    operands=[Operand('lhs'), Operand('rhs')],
    xls_type_expression='lhs->GetType()',
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')]
)

OpClass.kinds['ARITH_OP'] = OpClass(
    name='ArithOp',
    op='op',
    operands=[Operand('lhs'), Operand('rhs')],
    xls_type_expression='function->package()->GetBitsType(width)',
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')],
    attributes=[Int64Attribute('width')]
)

OpClass.kinds['BITWISE_REDUCTION_OP'] = OpClass(
    name='BitwiseReductionOp',
    op='op',
    operands=[Operand('operand')],
    xls_type_expression='function->package()->GetBitsType(1)',
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')]
)

OpClass.kinds['NARY_OP'] = OpClass(
    name='NaryOp',
    op='op',
    operands=[OperandSpan('args')],
    xls_type_expression='args[0]->GetType()',
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')]
)

OpClass.kinds['BIT_SLICE'] = OpClass(
    name='BitSlice',
    op='Op::kBitSlice',
    operands=[Operand('arg')],
    xls_type_expression='function->package()->GetBitsType(width)',
    attributes=[Int64Attribute('start'),
                Int64Attribute('width')],
)

OpClass.kinds['DYNAMIC_BIT_SLICE'] = OpClass(
    name='DynamicBitSlice',
    op='Op::kDynamicBitSlice',
    operands=[Operand('arg'), Operand('start')],
    xls_type_expression='function->package()->GetBitsType(width)',
    attributes=[Int64Attribute('width')],
)

OpClass.kinds['COMPARE_OP'] = OpClass(
    name='CompareOp',
    op='op',
    operands=[Operand('lhs'), Operand('rhs')],
    xls_type_expression='function->package()->GetBitsType(1)',
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')]
)

OpClass.kinds['CONCAT'] = OpClass(
    name='Concat',
    op='Op::kConcat',
    operands=[OperandSpan('args')],
    xls_type_expression='GetConcatType(function->package(), args)',
    extra_methods=[
        Method(name='GetOperandSliceData', return_cpp_type='SliceData',
               expression=None, params='int64 operandno'),
    ],
)

OpClass.kinds['COUNTED_FOR'] = OpClass(
    name='CountedFor',
    op='Op::kCountedFor',
    operands=[Operand('initial_value'),
              OperandSpan('invariant_args')],
    xls_type_expression='initial_value->GetType()',
    attributes=[Int64Attribute('trip_count'),
                Int64Attribute('stride'),
                FunctionAttribute('body')],
    extra_methods=[Method(name='initial_value',
                          return_cpp_type='Node*',
                          expression='operand(0)'),
                   Method(name='invariant_args',
                          return_cpp_type='absl::Span<Node* const>',
                          expression='operands().subspan(1)')],
    custom_clone_method=True
)

OpClass.kinds['EXTEND_OP'] = OpClass(
    name='ExtendOp',
    op='op',
    operands=[Operand('arg')],
    xls_type_expression='function->package()->GetBitsType(new_bit_count)',
    attributes=[Int64Attribute('new_bit_count')],
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')]
)

OpClass.kinds['INVOKE'] = OpClass(
    name='Invoke',
    op='Op::kInvoke',
    operands=[OperandSpan('args')],
    xls_type_expression='to_apply->return_value()->GetType()',
    attributes=[FunctionAttribute('to_apply')],
)

OpClass.kinds['LITERAL'] = OpClass(
    name='Literal',
    op='Op::kLiteral',
    operands=[],
    xls_type_expression='function->package()->GetTypeForValue(value)',
    attributes=[ValueAttribute('value')],
    extra_methods=[Method('IsZero', 'bool',
                          'value().IsBits() && value().bits().IsAllZeros()')],
)

OpClass.kinds['MAP'] = OpClass(
    name='Map',
    op='Op::kMap',
    operands=[Operand('arg')],
    xls_type_expression='GetMapType(arg, to_apply)',
    attributes=[FunctionAttribute('to_apply')],
)

OpClass.kinds['ONE_HOT'] = OpClass(
    name='OneHot',
    op='Op::kOneHot',
    attributes=[LsbOrMsbAttribute('priority')],
    operands=[Operand('input')],
    xls_type_expression='function->package()->GetBitsType('
    'input->BitCountOrDie() + 1)'
)

OpClass.kinds['ONE_HOT_SELECT'] = OpClass(
    name='OneHotSelect',
    op='Op::kOneHotSel',
    operands=[Operand('selector'),
              OperandSpan('cases')],
    xls_type_expression='cases[0]->GetType()',
    extra_methods=[Method(name='selector',
                          return_cpp_type='Node*',
                          expression='operand(0)'),
                   Method(name='cases',
                          return_cpp_type='absl::Span<Node* const>',
                          expression='operands().subspan(1)')],
    custom_clone_method=True
)

OpClass.kinds['PARAM'] = OpClass(
    name='Param',
    op='Op::kParam',
    operands=[],
    xls_type_expression='type',
    attributes=[StringAttribute('name')],
    extra_constructor_args=[ConstructorArgument(name='type',
                                                cpp_type='Type*',
                                                clone_expression='GetType()')],
)

OpClass.kinds['SELECT'] = OpClass(
    name='Select',
    op='Op::kSel',
    operands=[Operand('selector'),
              OperandSpan('cases'),
              OptionalOperand('default_value')],
    xls_type_expression='cases[0]->GetType()',
    extra_data_members=[
        DataMember(name='cases_size_',
                   cpp_type='int64',
                   init='cases.size()'),
        DataMember(name='has_default_value_',
                   cpp_type='bool',
                   init='default_value.has_value()')],
    extra_methods=[Method(name='selector',
                          return_cpp_type='Node*',
                          expression='operand(0)'),
                   Method(name='cases',
                          return_cpp_type='absl::Span<Node* const>',
                          expression='operands().subspan(1, cases_size_)'),
                   Method(name='default_value',
                          return_cpp_type='absl::optional<Node*>',
                          expression='has_default_value_ '
                          '? absl::optional<Node*>(operands().back()) '
                          ': absl::nullopt'),
                   Method(name='AllCases',
                          return_cpp_type='bool',
                          expression='',
                          params='std::function<bool(Node*)> p'),
                   Method(name='any_case',
                          return_cpp_type='Node*',
                          expression='!cases().empty() ? cases().front() : default_value().has_value() ? default_value().value() : nullptr')],
    custom_clone_method=True
)

OpClass.kinds['TUPLE'] = OpClass(
    name='Tuple',
    op='Op::kTuple',
    operands=[OperandSpan('elements')],
    xls_type_expression='GetTupleType(function->package(), elements)',
    extra_methods=[Method(name='size',
                          return_cpp_type='int64',
                          expression='operand_count()')],
)

OpClass.kinds['TUPLE_INDEX'] = OpClass(
    name='TupleIndex',
    op='Op::kTupleIndex',
    operands=[Operand('arg')],
    xls_type_expression='arg->GetType()->AsTupleOrDie()->element_type(index)',
    attributes=[Int64Attribute('index')],
)

OpClass.kinds['UN_OP'] = OpClass(
    name='UnOp',
    op='op',
    operands=[Operand('arg')],
    xls_type_expression='arg->GetType()',
    extra_constructor_args=[ConstructorArgument(name='op',
                                                cpp_type='Op',
                                                clone_expression='op()')]
)

OpClass.kinds['DECODE'] = OpClass(
    name='Decode',
    op='Op::kDecode',
    operands=[Operand('arg')],
    xls_type_expression='function->package()->GetBitsType(width)',
    attributes=[Int64Attribute('width')],
)

OpClass.kinds['ENCODE'] = OpClass(
    name='Encode',
    op='Op::kEncode',
    operands=[Operand('arg')],
    # Subtract one from the width expression to account for zero-based
    # numbering.
    xls_type_expression='function->package()->GetBitsType(Bits::MinBitCountUnsigned(arg->BitCountOrDie() - 1))',
)

OPS = [
    Op(
        enum_name='kAdd',
        name='add',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[Property.ASSOCIATIVE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kAnd',
        name='and',
        op_class=OpClass.kinds['NARY_OP'],
        properties=[Property.BITWISE,
                    Property.ASSOCIATIVE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kAndReduce',
        name='and_reduce',
        op_class=OpClass.kinds['BITWISE_REDUCTION_OP'],
        properties=[],
    ),
    Op(
        enum_name='kNand',
        name='nand',
        op_class=OpClass.kinds['NARY_OP'],
        # Note: not associative, because of the inversion.
        properties=[Property.BITWISE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kNor',
        name='nor',
        op_class=OpClass.kinds['NARY_OP'],
        # Note: not associative, because of the inversion.
        properties=[Property.BITWISE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kArray',
        name='array',
        op_class=OpClass.kinds['ARRAY'],
        properties=[],
    ),
    Op(
        enum_name='kArrayIndex',
        name='array_index',
        op_class=OpClass.kinds['ARRAY_INDEX'],
        properties=[],
    ),
    Op(
        enum_name='kArrayUpdate',
        name='array_update',
        op_class=OpClass.kinds['ARRAY_UPDATE'],
        properties=[],
    ),
    Op(
        enum_name='kBitSlice',
        name='bit_slice',
        op_class=OpClass.kinds['BIT_SLICE'],
        properties=[],
    ),
    Op(
        enum_name='kDynamicBitSlice',
        name='dynamic_bit_slice',
        op_class=OpClass.kinds['DYNAMIC_BIT_SLICE'],
        properties=[],
    ),
    Op(
        enum_name='kConcat',
        name='concat',
        op_class=OpClass.kinds['CONCAT'],
        properties=[],
    ),
    Op(
        enum_name='kCountedFor',
        name='counted_for',
        op_class=OpClass.kinds['COUNTED_FOR'],
        properties=[],
    ),
    Op(
        enum_name='kDecode',
        name='decode',
        op_class=OpClass.kinds['DECODE'],
        properties=[],
    ),
    Op(
        enum_name='kEncode',
        name='encode',
        op_class=OpClass.kinds['ENCODE'],
        properties=[],
    ),
    Op(
        enum_name='kEq',
        name='eq',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kIdentity',
        name='identity',
        op_class=OpClass.kinds['UN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kInvoke',
        name='invoke',
        op_class=OpClass.kinds['INVOKE'],
        properties=[],
    ),
    Op(
        enum_name='kLiteral',
        name='literal',
        op_class=OpClass.kinds['LITERAL'],
        properties=[],
    ),
    Op(
        enum_name='kMap',
        name='map',
        op_class=OpClass.kinds['MAP'],
        properties=[],
    ),
    Op(
        enum_name='kNe',
        name='ne',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kNeg',
        name='neg',
        op_class=OpClass.kinds['UN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kNot',
        name='not',
        op_class=OpClass.kinds['UN_OP'],
        properties=[Property.BITWISE],
    ),
    Op(
        enum_name='kOneHot',
        name='one_hot',
        op_class=OpClass.kinds['ONE_HOT'],
        properties=[],
    ),
    Op(
        enum_name='kOneHotSel',
        name='one_hot_sel',
        op_class=OpClass.kinds['ONE_HOT_SELECT'],
        properties=[],
    ),
    Op(
        enum_name='kOr',
        name='or',
        op_class=OpClass.kinds['NARY_OP'],
        properties=[Property.BITWISE,
                    Property.ASSOCIATIVE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kOrReduce',
        name='or_reduce',
        op_class=OpClass.kinds['BITWISE_REDUCTION_OP'],
        properties=[],
    ),
    Op(
        enum_name='kParam',
        name='param',
        op_class=OpClass.kinds['PARAM'],
        properties=[],
    ),
    Op(
        enum_name='kReverse',
        name='reverse',
        op_class=OpClass.kinds['UN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kSDiv',
        name='sdiv',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kSel',
        name='sel',
        op_class=OpClass.kinds['SELECT'],
        properties=[],
    ),
    Op(
        enum_name='kSGe',
        name='sge',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kSGt',
        name='sgt',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kShll',
        name='shll',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kShrl',
        name='shrl',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kShra',
        name='shra',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kSignExt',
        name='sign_ext',
        op_class=OpClass.kinds['EXTEND_OP'],
        properties=[],
    ),
    Op(
        enum_name='kSLe',
        name='sle',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kSLt',
        name='slt',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kSMul',
        name='smul',
        op_class=OpClass.kinds['ARITH_OP'],
        properties=[Property.ASSOCIATIVE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kSub',
        name='sub',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kTuple',
        name='tuple',
        op_class=OpClass.kinds['TUPLE'],
        properties=[],
    ),
    Op(
        enum_name='kTupleIndex',
        name='tuple_index',
        op_class=OpClass.kinds['TUPLE_INDEX'],
        properties=[],
    ),
    Op(
        enum_name='kUDiv',
        name='udiv',
        op_class=OpClass.kinds['BIN_OP'],
        properties=[],
    ),
    Op(
        enum_name='kUGe',
        name='uge',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kUGt',
        name='ugt',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kULe',
        name='ule',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kULt',
        name='ult',
        op_class=OpClass.kinds['COMPARE_OP'],
        properties=[Property.COMPARISON],
    ),
    Op(
        enum_name='kUMul',
        name='umul',
        op_class=OpClass.kinds['ARITH_OP'],
        properties=[Property.ASSOCIATIVE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kXor',
        name='xor',
        op_class=OpClass.kinds['NARY_OP'],
        properties=[Property.BITWISE,
                    Property.ASSOCIATIVE,
                    Property.COMMUTATIVE],
    ),
    Op(
        enum_name='kXorReduce',
        name='xor_reduce',
        op_class=OpClass.kinds['BITWISE_REDUCTION_OP'],
        properties=[],
    ),
    Op(
        enum_name='kZeroExt',
        name='zero_ext',
        op_class=OpClass.kinds['EXTEND_OP'],
        properties=[],
    ),
]
# pyformat: enable
