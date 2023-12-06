# Copyright 2023 The XLS Authors
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

"""Rule for stripping comments out of DSLX files present in a filegroup.

This is useful e.g. when we create a corpus of `.x` files to present to fuzzing
infrastructure -- directories will generally present a filegroup filled with
comment-stripped `.x` files.
"""

def _dslx_strip_comments_impl(ctx):
    outs = []
    for src in ctx.files.srcs:
        out_file = ctx.actions.declare_file("%s.stripped.x" % src.basename)
        outs.append(out_file)
        args = ctx.actions.args()
        args.add("--original_on_error")
        args.add(src.path)
        args.add("--output_path")
        args.add(out_file.path)
        ctx.actions.run(
            outputs = [out_file],
            inputs = [src] + [ctx.executable.strip_comments],
            executable = ctx.executable.strip_comments,
            arguments = [args],
            progress_message = "Stripping comments from DSLX file %{input} to create %{output}",
        )
    runfiles = ctx.runfiles(files = outs)
    return [DefaultInfo(runfiles = runfiles)]

dslx_strip_comments = rule(
    implementation = _dslx_strip_comments_impl,
    attrs = {
        "srcs": attr.label_list(),
        "strip_comments": attr.label(
            default = Label("//xls/dslx:strip_comments_main"),
            executable = True,
            cfg = "exec",
            allow_single_file = True,
        ),
    },
)
