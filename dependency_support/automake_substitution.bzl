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

"""Provides helper that replaces @VARIABLE_NAME@ occurences with values, as
specified by a provided map."""

def automake_substitution(name, src, out, substitutions = {}):
    """Replaces @VARIABLE_NAME@ occurences with values.

    Note: The current implementation does not allow slashes in variable
    values."""

    substitution_pipe = " ".join([
        "| sed 's/@%s@/%s/g'" % (variable_name, substitutions[variable_name])
        for variable_name in substitutions.keys()
    ])
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = "cat $(location :%s) %s > $@" % (src, substitution_pipe),
    )
