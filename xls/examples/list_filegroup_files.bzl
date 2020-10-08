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

"""Contains a macro that writes the files in a filegroup to a file."""

def _list_filegroup_files(ctx):
    out = ctx.actions.declare_file(ctx.attr.out)

    # instead of generating a file, you could just print the info
    ctx.actions.write(
        output = out,
        content = "\n".join([f.short_path for f in ctx.files.src]),
    )
    return DefaultInfo(files = depset([out]))

list_filegroup_files = rule(
    implementation = _list_filegroup_files,
    attrs = {
        "src": attr.label(),
        "out": attr.string(),
    },
    outputs = {
        "out": "%{out}",
    },
)
