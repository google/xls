# Copyright 2020 The XLS Authors
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

"""Pretty printers for IR nodes and structures in XLS.

This makes LLDB print the 'ToString' representation of xls::Node, xls::Type,
xls::BValue, xls::Value, xls::Bits, xls::dslx::ASTNode, xls::dslx::Type,
xls::dslx::InterpValue, and xls::dslx::ParametricContext objects as the summary.
This greatly improves usability of the debugger when examining tests.

Usage:
  (lldb) command script import xls/dev_tools/lldbprettyprint/lldbprettyprint.py

NB No real effort was made to optimize these printers. Attempting to summarize a
large number of them might take noticeable time.

Since the ToString methods for these types won't crash hard enough to take out
LLDB only relatively minor efforts are made to prevent us from calling the
ToString methods on uninitialized variables. LLDB seems to handle this case
fine.
"""


def init_printers(debugger):
  """Initalize printers for ir structs."""
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_bvalue_summary xls::BValue"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_value_summary xls::Value"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_irnode_summary xls::Node"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_irtype_summary xls::Type"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_bits_summary xls::Bits"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_interval_summary xls::Interval"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_interval_set_summary xls::IntervalSet"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_astnode_summary --recognizer-function"
      " lldbprettyprint.xlsir.xls_is_astnode"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_dslxtype_summary --recognizer-function"
      " lldbprettyprint.xlsir.xls_is_dslxtype"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_interpvalue_summary xls::dslx::InterpValue"
  )
  debugger.HandleCommand(
      "type summary add --expand --python-function"
      " lldbprettyprint.xlsir.xls_parametriccontext_summary"
      " xls::dslx::ParametricContext"
  )


def xls_bvalue_summary(valobj, _):
  """Summarize xls::BValue."""
  node = valobj.GetChildMemberWithName("node_")
  if not valobj.IsInScope():
    return "<uninitialized>"
  if not node.IsValid() or node.GetValueAsUnsigned() == 0:
    return "<Null BValue>"
  else:
    return node.GetSummary()


def _maybe_deref(v):
  """Standardize a value to an & ref."""
  if v.TypeIsPointerType():
    return v.Dereference()
  else:
    return v


def xls_irtype_summary(valobj, _):
  """Summarize xls::Type."""
  # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
  # to do this stuff is implement it all in python. This seems to be fast enough
  # however.
  # TODO(allight): Consider rewriting in python.
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_irnode_summary(valobj, _):
  """Summarize xls::Node."""
  # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
  # to do this stuff is implement it all in python. This seems to be fast enough
  # however.
  # TODO(allight): Consider rewriting in python.
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_value_summary(valobj, _):
  """Summarize xls::Value."""
  # Check for validity (not really needed just cleans up output a bit)
  try:
    valobj = _maybe_deref(valobj)
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    kind = valobj.GetChildMemberWithName("kind_")
    kind_type = kind.GetType()
    if not any(
        kind.GetValueAsSigned() == m.GetValueAsSigned()
        for m in kind_type.enum_members
    ):
      return "<unintiailized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        valobj.EvaluateExpression(
            "this.ToString(xls::FormatPreference::kDefault)"
        )
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_bits_summary(valobj, _):
  """Summarize xls::Bits."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToDebugString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_interval_summary(valobj, _):
  """Summarize xls::Interval."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_interval_set_summary(valobj, _):
  """Summarize xls::IntervalSet."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def base_type_recognizer_function_generator(base_name):
  def is_base_type(sbtype, _):
    if sbtype.IsReferenceType():
      sbtype = sbtype.GetDereferencedType()
    if sbtype.IsPointerType():
      sbtype = sbtype.GetPointeeType()
    for base in sbtype.get_bases_array():
      if base.GetName() == base_name or is_base_type(base.GetType(), _):
        return True
    return False

  return is_base_type


xls_is_astnode = base_type_recognizer_function_generator("xls::dslx::AstNode")


def xls_astnode_summary(valobj, _):
  """Summarize xls::dslx::AstNode."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


xls_is_dslxtype = base_type_recognizer_function_generator("xls::dslx::Type")


def xls_dslxtype_summary(valobj, _):
  """Summarize xls::dslx::Type."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_interpvalue_summary(valobj, _):
  """Summarize xls::dslx::InterpValue."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression(
            "this.ToStringInternal(false, xls::FormatPreference::kDefault)"
        )
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"


def xls_parametriccontext_summary(valobj, _):
  """Summarize xls::dslx::ParametricContext."""
  try:
    if not valobj.IsValid() or not valobj.IsInScope():
      return "<uninitialized>"
    # Using EvaluateExpression like this can be somewhat slow. The 'correct' way
    # to do this stuff is implement it all in python. This seems to be fast
    # enough however.
    # TODO(allight): Consider rewriting in python.
    return (
        _maybe_deref(valobj)
        .EvaluateExpression("this.ToString()")
        .GetSummary()
        .strip('"')
    )
  # pylint: disable-next=broad-exception-caught
  except Exception:
    return "<INVALID>"
