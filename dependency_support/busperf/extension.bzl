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

"""Module extension for busperf."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _busperf_extension_impl(
        module_ctx):  # @unused
    http_archive(
        name = "busperf",
        urls = ["https://github.com/antmicro/busperf/archive/a62c4c7c6cb3edb090a6b17f0c6edf18fcb4ebb1.tar.gz"],
        integrity = "sha256-tfu+WqRT05t2SsMy6DjY+142ZlkqbzMc9fWIfzzpsI4=",
        strip_prefix = "busperf-a62c4c7c6cb3edb090a6b17f0c6edf18fcb4ebb1",
        build_file = Label("//dependency_support/busperf:bundled.BUILD.bazel"),
    )

busperf_extension = module_extension(
    implementation = _busperf_extension_impl,
)

def _rust_toolchain_repository_impl(repository_ctx):
    # Self-contained rustup toolchain, with wasm32-unknown-unknown for
    # busperf's HTML report build.
    #
    # TODO: temporary workaround, easiest way to get the wasm build working.
    # Should be replaced with a proper rules_rust build.
    rustup_init = repository_ctx.path("rustup-init")
    repository_ctx.download(
        url = "https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init",
        output = rustup_init,
        executable = True,
    )

    rustup_home = repository_ctx.path("rustup")
    cargo_home = repository_ctx.path("cargo")
    result = repository_ctx.execute(
        [
            str(rustup_init),
            "-y",
            "--no-modify-path",
            "--profile",
            "minimal",
            "--default-toolchain",
            "stable",
            "--target",
            "wasm32-unknown-unknown",
        ],
        environment = {
            "RUSTUP_HOME": str(rustup_home),
            "CARGO_HOME": str(cargo_home),
        },
        timeout = 1800,
    )
    if result.return_code != 0:
        fail("rustup-init failed:\n" + result.stdout + result.stderr)

    repository_ctx.file("BUILD.bazel", """\
exports_files(["cargo/bin/cargo"])
""")

rust_toolchain_repository = repository_rule(
    implementation = _rust_toolchain_repository_impl,
)

def _rust_toolchain_extension_impl(
        module_ctx):  # @unused
    rust_toolchain_repository(name = "busperf_rust_toolchain")

rust_toolchain_extension = module_extension(
    implementation = _rust_toolchain_extension_impl,
)
