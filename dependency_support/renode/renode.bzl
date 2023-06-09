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

load("@renode_robot_deps//:requirements.bzl", "requirement")

def renode_test_impl(ctx):
    toolchain = ctx.toolchains["@renode//:toolchain_type"].renode_runtime
    toolchain_files = depset(toolchain.runtime)

    py_toolchain = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"].py3_runtime

    wrapper = ctx.actions.declare_file('wrapper.sh')

    script = "RENODE_CI_MODE=YES {} \
--renode-config $TEST_UNDECLARED_OUTPUTS_DIR/renode_config \
-r $TEST_UNDECLARED_OUTPUTS_DIR {}".format(
        toolchain.renode_test.path,
        " ".join([file.short_path for file in ctx.files.robot])
    )
    ctx.actions.write(
        output = wrapper,
        content = script,
    )

    runfiles = ctx.runfiles(
        files = toolchain.runtime + ctx.files.robot,
        transitive_files=py_toolchain.files
    ).merge_all(
      [dep.default_runfiles for dep in ctx.attr._default_reqs] +
      [dep.default_runfiles for dep in ctx.attr.pip_reqs]
    )

    modules_roots = [
        "./" + repo.label.workspace_root + "/" + repo.label.package
        for repo in (ctx.attr.pip_reqs + ctx.attr._default_reqs)
    ]
    pypath = ":".join([str(p) for p in modules_roots])
    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH" : pypath,
    }

    return [
        DefaultInfo(executable = wrapper, runfiles = runfiles),
        testing.TestEnvironment(env),
    ]



renode_test = rule(
    implementation = renode_test_impl,
    attrs = {
        "robot": attr.label_list(allow_empty = False, mandatory = True, allow_files=True),
        "pip_reqs": attr.label_list(default = [], allow_files=True),
        "_default_reqs": attr.label_list(default = [
            requirement("robotframework"),
            requirement("requests"),
            requirement("psutil"),
            requirement("pyyaml"),
        ]),
    },
    toolchains = [
        "@renode//:toolchain_type",
        "@bazel_tools//tools/python:toolchain_type",
    ],
    test = True,
)
