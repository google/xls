# Copyright 2020 Google LLC
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

"""Generates a fuzz issue regression suite."""

def _to_suffix(path):
    basename = path.split("/")[-1]
    if not basename.endswith(".x"):
        fail()
    basename = basename[:-len(".x")]
    if basename.startswith("crasher_"):
        basename = basename[len("crasher_"):]
    return basename

def generate_crasher_regression_tests(srcs, prefix, failing = None, optonly = None):
    """Generates targets for fuzz-found issues.

    Also generates a manual suite called ":regression_tests" that will include
    known-failing targets.

    Args:
        srcs: Testdata files (DSLX) to run regressions for.
        prefix: Prefix directory path for the testdata files.
        failing: Optional list of failing testdata paths (must be a string
            match with a member in srcs).
        optonly: Optional list of testdata paths to only run in opt mode (must
            be a string match with a member in srcs).
    """
    failing = failing or []
    optonly = optonly or []
    names = []
    for f in srcs:
        if not f.endswith(".x"):
            fail()

        name = "run_crasher_test_{}".format(_to_suffix(f))
        names.append(name)
        fullpath = prefix + "/" + f
        broken = f in failing
        is_optonly = f in optonly
        native.sh_test(
            name = name,
            srcs = ["//xls/dslx/fuzzer:run_crasher_sh"],
            args = [fullpath],
            data = [
                "//xls/dslx/fuzzer:run_crasher",
                f,
            ],
            tags = (
                (["broken", "manual"] if broken else []) +
                (["optonly"] if is_optonly else [])
            ),
        )

    native.test_suite(
        name = "regression_tests",
        tests = names,
        tags = ["manual"],
    )
