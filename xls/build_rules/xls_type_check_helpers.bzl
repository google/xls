# Copyright 2022 The XLS Authors
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

"""This module contains type check helpers.

These functions are typically used to validate the arguments of macro definitions.
"""

def _type_check(arg_name, arg_value, subject_type_name, subject_type, can_be_none = False):
    """A macro that produces a failure if the type of 'arg_value' is not of type 'subject_type'.

    Args:
      arg_name: The name of the argument. It is used in the failure message to reference the
        argument.
      arg_value: The value of the argument.
      subject_type_name: The name of the subject type.
      subject_type: The type of the subject.
      can_be_none: Flag denoting that the type of 'arg_valu'e can also be of type 'None'.
    """
    if can_be_none and not arg_value:
        return
    if (type(arg_value) != subject_type):
        fail("Argument '%s' must be of %s type." % arg_name, subject_type_name)

def bool_type_check(argument_name, argument_value, can_be_none = False):
    """A macro that produces a failure if the value is not of bool type.

    Args:
      argument_name: The name of the argument. It is used in the failure message to reference the
        argument.
      argument_value: The value of the argument.
      can_be_none: Flag denoting that the type of the argument value can also be of type 'None'.
    """
    _type_check(argument_name, argument_value, "bool", type(True), can_be_none)

def int_type_check(argument_name, argument_value, can_be_none = False):
    """A macro that produces a failure if the value is not of integer type.

    Args:
      argument_name: The name of the argument. It is used in the failure message to reference the
        argument.
      argument_value: The value of the argument.
      can_be_none: Flag denoting that the type of the argument value can also be of type 'None'.
    """
    _type_check(argument_name, argument_value, "integer", type(0), can_be_none)

def string_type_check(argument_name, argument_value, can_be_none = False):
    """A macro that produces a failure if the value is not of string type.

    Args:
      argument_name: The name of the argument. It is used in the failure message to reference the
        argument.
      argument_value: The value of the argument.
      can_be_none: Flag denoting that the type of the argument value can also be of type 'None'.
    """
    _type_check(argument_name, argument_value, "string", type(""), can_be_none)

def list_type_check(argument_name, argument_value, can_be_none = False):
    """A macro that produces a failure if the value is not of list type.

    Args:
      argument_name: The name of the argument. It is used in the failure message to reference the
        argument.
      argument_value: The value of the argument.
      can_be_none: Flag denoting that the type of the argument value can also be of type none.
    """
    _type_check(argument_name, argument_value, "list", type([]), can_be_none)

def dictionary_type_check(argument_name, argument_value, can_be_none = False):
    """A macro that produces a failure if the value is not of dictionary type.

    Args:
      argument_name: The name of the argument. It is used in the failure message to reference the
        argument.
      argument_value: The value of the argument.
      argument_value: The value of the argument.
      can_be_none: Flag denoting that the type of the argument value can also be of type 'None'.
    """
    _type_check(argument_name, argument_value, "dictionary", type({}), can_be_none)

def tuple_type_check(argument_name, argument_value, can_be_none = False):
    """A macro that produces a failure if the value is not of tuple type.

    Args:
      argument_name: The name of the argument. It is used in the failure message to reference the
        argument.
      argument_value: The value of the argument.
      argument_value: The value of the argument.
      can_be_none: Flag denoting that the type of the argument value can also be of type 'None'.
    """
    _type_check(argument_name, argument_value, "tuple", "tuple", can_be_none)
