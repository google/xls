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

"""Utilities for building libedit."""

def _generated_headers(names):
    """Transforms names into their generated equivalent."""
    return [":src/%s.h" % name for name in names]

def _makelist(name, srcs, flag):
    """Runs makelist over a set of inputs to generate a header file."""
    native.genrule(
        name = "%s_makelist" % name,
        srcs = srcs,
        outs = ["src/%s.h" % name],
        tools = ["src/makelist"],
        cmd = "sh $(location src/makelist) %s $(SRCS) > $@" % flag,
    )

# The base files for makelist calls.
_inputs = ["common", "emacs", "vi"]

# The headers generated directly from inputs.
_input_headers = _generated_headers(_inputs)

# The full set of headers generated, used for srcs.
makelist_headers = _input_headers + _generated_headers(["fcns", "func", "help"])

def makelist_genrules():
    """Runs all necessary makelist calls."""
    for name in _inputs:
        _makelist(name, ["src/%s.c" % name], "-h")
    _makelist("fcns", _input_headers, "-fh")
    _makelist("func", _input_headers, "-fc")
    _makelist("help", ["src/%s.c" % name for name in _inputs], "-bh")
