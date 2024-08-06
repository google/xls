# Copyright 2024 The XLS Authors
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

"""A few helper rules for interacting directly with xls_aot_generate rules.

Since this directory includes the internal implementation of these rules it
alone sometimes needs to reach into them in ways other tools do not.
"""

load("//xls/build_rules:xls_providers.bzl", "AotCompileInfo")

def _aot_protobuf(ctx):
    """
    Make a depset with only the protobuf compile info for genrules and such to examine.
    """
    return DefaultInfo(files = depset([ctx.attr.aot[AotCompileInfo].proto_file]))

aot_protobuf = rule(implementation = _aot_protobuf, attrs = {
    "aot": attr.label(
        doc = "xls_aot_generate target to get the protobuf from",
        providers = [AotCompileInfo],
    ),
})
