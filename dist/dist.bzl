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

"""Provides utility rules for naming release artifacts."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_pkg//pkg:providers.bzl", "PackageVariablesInfo")

def _xls_release(ctx):
    values = {
        "version": ctx.attr.version[BuildSettingInfo].value,
        "os": ctx.attr.os[BuildSettingInfo].value,
        "arch": ctx.attr.arch[BuildSettingInfo].value,
    }
    return PackageVariablesInfo(values = values)

xls_release = rule(
    implementation = _xls_release,
    attrs = {
        "version": attr.label(),
        "os": attr.label(),
        "arch": attr.label(),
    },
)
