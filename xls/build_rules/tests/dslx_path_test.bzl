# Copyright 2026 The XLS Authors
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

"""Unit tests for build_dslx_path in xls_dslx_common.bzl."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//xls/build_rules:xls_dslx_common.bzl", "build_dslx_path")

def _fake_ctx(genfiles = "bazel-out/k8-fastbuild/genfiles", bin_dir = "bazel-out/k8-fastbuild/bin"):
    return struct(
        genfiles_dir = struct(path = genfiles),
        bin_dir = struct(path = bin_dir),
    )

def _fake_label(workspace_name = "", workspace_root = ""):
    return struct(
        workspace_name = workspace_name,
        workspace_root = workspace_root,
    )

def _fake_file(root_path = "", workspace_name = "", workspace_root = ""):
    return struct(
        root = struct(path = root_path),
        owner = _fake_label(workspace_name = workspace_name, workspace_root = workspace_root),
    )

def _root_module_build_action_test_impl(ctx):
    env = unittest.begin(ctx)
    path = build_dslx_path(
        ctx = _fake_ctx(),
        tool_label = _fake_label(),
        dslx_srcs = [_fake_file(), _fake_file()],
        is_test = False,
    )
    entries = path.split(":")
    asserts.equals(
        env,
        [".", "bazel-out/k8-fastbuild/genfiles", "bazel-out/k8-fastbuild/bin"],
        entries,
    )
    return unittest.end(env)

def _child_module_test_action_test_impl(ctx):
    env = unittest.begin(ctx)
    path = build_dslx_path(
        ctx = _fake_ctx(),
        tool_label = _fake_label(workspace_name = "xls+", workspace_root = "external/xls+"),
        dslx_srcs = [
            _fake_file(),
            _fake_file(workspace_name = "other_dep+", workspace_root = "external/other_dep+"),
        ],
        is_test = True,
    )
    entries = path.split(":")
    asserts.true(env, "${PWD}" in entries, "Expected ${PWD} in test dslx_path")
    asserts.true(
        env,
        "../xls+" in entries,
        "Expected runfiles-relative ../xls+ for external XLS tool in test dslx_path: " + path,
    )
    asserts.true(
        env,
        "../other_dep+" in entries,
        "Expected runfiles-relative ../other_dep+ for external DSLX dep in test dslx_path: " + path,
    )

    # Runfiles-relative ../<repo> should appear before execroot external/<repo>.
    asserts.true(
        env,
        entries.index("../xls+") < entries.index("external/xls+"),
        "Expected ../xls+ before external/xls+ in: " + path,
    )
    asserts.true(
        env,
        entries.index("../other_dep+") < entries.index("external/other_dep+"),
        "Expected ../other_dep+ before external/other_dep+ in: " + path,
    )
    return unittest.end(env)

def _child_module_build_action_with_generated_external_src_test_impl(ctx):
    env = unittest.begin(ctx)
    path = build_dslx_path(
        ctx = _fake_ctx(),
        tool_label = _fake_label(workspace_name = "xls+", workspace_root = "external/xls+"),
        dslx_srcs = [
            _fake_file(
                root_path = "bazel-out/k8-opt/bin",
                workspace_name = "other_dep+",
                workspace_root = "external/other_dep+",
            ),
        ],
        user_dslx_path = "custom/one:custom/two:custom/one",
        is_test = False,
    )
    entries = path.split(":")
    asserts.false(env, "${PWD}" in entries, "Build action must not emit literal ${PWD}: " + path)
    asserts.false(env, "../xls+" in entries, "Build action must not emit test-only ../xls+: " + path)
    asserts.equals(env, "custom/one", entries[0])
    asserts.equals(env, "custom/two", entries[1])
    asserts.equals(env, ".", entries[2])
    asserts.true(env, "external/other_dep+" in entries)
    asserts.true(env, "bazel-out/k8-opt/bin/external/other_dep+" in entries)
    asserts.true(env, "external/xls+" in entries)

    # Verify deduplication (custom/one only once).
    asserts.equals(env, len(entries), len({e: True for e in entries}))
    return unittest.end(env)

root_module_build_action_test = unittest.make(_root_module_build_action_test_impl)
child_module_test_action_test = unittest.make(_child_module_test_action_test_impl)
child_module_build_action_with_generated_external_src_test = unittest.make(
    _child_module_build_action_with_generated_external_src_test_impl,
)

def dslx_path_test_suite(name):
    unittest.suite(
        name,
        root_module_build_action_test,
        child_module_test_action_test,
        child_module_build_action_with_generated_external_src_test,
    )
