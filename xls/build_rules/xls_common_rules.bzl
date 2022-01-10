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

"""This module contains helpers for XLS build rules."""

def get_args(arguments, valid_arguments, default_arguments = {}):
    """Returns a string representation of the arguments.

    The macro builds a string representation of the arguments. If no value is
    specified in the arguments, the a default value will be selected (if
    present). If an argument in the list is not an acceptable argument, an error
    is thrown.

    Example:
      1) An unspecified a default argument value.
        Input:
          arguments = {"argument1": "42", "argument2": "binary"}
          valid_arguments = {"argument1", "argument2", "arguments3"}
          default_arguments = {"arguments3" : "foo"}
          _get_args(arguments, valid_arguments, default_arguments)

        Output:
            --argument1=42 --argument2=binary --argument3=foo
      2) Overriding a default argument value.
        Input:
          arguments = {"argument1": "42", "argument2": "binary"}
          valid_arguments = {"argument1", "argument2", "arguments3"}
          default_arguments = {"argument1" : "0", "arguments3" : "foo"}
          _get_args(arguments, valid_arguments, default_arguments)

        Output:
            --argument1=42 --argument2=binary --argument3=foo
      3) An invalid argument.
        Input:
          arguments = {"argument1": "42", "argument2": "binary"}
          valid_arguments = {"argument1"}
          default_arguments = {}
          _get_args(arguments, valid_arguments, default_arguments)

        Output (error with message):
            Unrecognized argument: argument2.

    Args:
      arguments: The list of arguments values.
      valid_arguments: The source file.
      default_arguments: A list of default argument values.
    Returns:
      A string of the arguments.
    """

    # Add arguments
    my_args = ""
    for flag_name in arguments:
        if flag_name not in valid_arguments:
            fail("Unrecognized argument: %s." % flag_name)
        my_args += " --%s=%s" % (flag_name, arguments[flag_name])

    # Add default arguments
    for flag_name in default_arguments:
        if flag_name not in valid_arguments:
            fail("Unrecognized argument: %s." % flag_name)
        if flag_name not in arguments:
            my_args += " --%s=%s" % (flag_name, default_arguments[flag_name])
    return my_args

def append_cmd_line_args_to(cmd):
    """Appends the syntax for command line arguments ("$*") to the cmd

    Args:
      cmd: A string representing the command.

    Returns:
      'cmd' with the syntax for command line arguments ("$*") appended
    """
    return cmd + " $*"

def get_output_filename_value(ctx, output_attr_name, default_filename):
    """Returns the filename for an output attribute within the context.

    If the output_attr_name is defined in the context, the function returns
    the value of the output_attr_name. Otherwise, the function returns the
    default_filename.

    Args:
      ctx: The current rule's context object.
      output_attr_name: The name of the output attribute.
      default_filename: The default filename.

    Returns:
      The filename for an output attribute within the context.
    """
    attribute_value = getattr(ctx.attr, output_attr_name)
    if attribute_value:
        return attribute_value.name
    else:
        return default_filename

def split_filename(filename):
    """Returns the basename and extension of a given filename.

    The basename and extension are distinguished using the rightmost dot (.).
    For example, "a.file.name.ext" the basename is equivalent to "a.file.name"
    and the extension is equivalent to "ext". If there is no dot in the
    filename, the basename is equivalent to the filename, and the extension is
    equivalent to 'None'.

    Args:
      filename: The filename.

    Returns:
      The filename for an output attribute within the context.
    """
    if "." not in filename:
        return [filename, None]
    return filename.rsplit(".", 1)
