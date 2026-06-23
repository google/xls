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

"""Module extension for Bitwuzla and its non-BCR dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Replace "config.h" include paths with relative paths to src/config.h, to avoid conflicts with
# config.h files from other packages (e.g., gmp).
_CONFIG_PATCH_CMD = """
  find . -type f \\( -name "*.cpp" -o -name "*.h" \\) | while read -r file; do
    # Avoid self-patching config.h
    if [ "$file" = "./src/config.h" ]; then continue; fi
  
    if grep -q '#include "config.h"' "$file"; then
      # Construct the relative path to the repository root:
      #   strip "./", then create a string with a "../" for each slash in the path.
      clean_path=${file#./}
      relative_root=$(echo "$clean_path" | tr -cd '/' | sed 's|/|../|g')
      
      sed 's|#include "config.h"|#include "'"${relative_root}"'src/config.h"|g' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
  done
"""

def _bitwuzla_extension_impl(module_ctx):  # @unused
    # CaDiCaL SAT solver
    http_archive(
        name = "cadical",
        urls = ["https://github.com/arminbiere/cadical/archive/3ff42f04384489916f017acd6d5e7cbfa7257be7.tar.gz"],
        strip_prefix = "cadical-3ff42f04384489916f017acd6d5e7cbfa7257be7",
        integrity = "sha256-+w7OK9Q2I6/HOPqUXFfRMM7ipu9lFoNcUQWoUDGSogU=",
        build_file = Label("//dependency_support/bitwuzla:cadical.BUILD.bazel"),
    )

    # SymFPU floating-point utility library
    http_archive(
        name = "symfpu",
        urls = ["https://github.com/martin-cs/symfpu/archive/8fbe139bf0071cbe0758d2f6690a546c69ff0053.tar.gz"],
        strip_prefix = "symfpu-8fbe139bf0071cbe0758d2f6690a546c69ff0053",
        integrity = "sha256-2guu77hPsArUStaZPZfC5sw1aPmu0NnoaTqmtZl1Lls=",
        build_file = Label("//dependency_support/bitwuzla:symfpu.BUILD.bazel"),
    )

    # Bitwuzla solver
    http_archive(
        name = "bitwuzla",
        urls = ["https://github.com/bitwuzla/bitwuzla/archive/3f5d9cd11dda80626b23bc2b353768abd6906f58.tar.gz"],
        strip_prefix = "bitwuzla-3f5d9cd11dda80626b23bc2b353768abd6906f58",
        build_file = Label("//dependency_support/bitwuzla:bundled.BUILD.bazel"),
        integrity = "sha256-8EoJYNCds0Fkuw9jKEHL47LsdCJ9d03750w+TFHSquA=",
        patches = [
            Label("//dependency_support/bitwuzla:config.patch"),
            Label("//dependency_support/bitwuzla:msan.patch"),
        ],
        patch_args = ["-p1"],
        patch_cmds = [_CONFIG_PATCH_CMD],
    )

bitwuzla_extension = module_extension(
    implementation = _bitwuzla_extension_impl,
)
