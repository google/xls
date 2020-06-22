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

def generate_crasher_regression_tests(name, srcs, prefix, failing = None, tags = None):
    """Generates targets for fuzz-found issues.

    Also generates a manual suite called ":regression_tests" that will include
    known-failing targets.

    Args:
        name: Name to use for the resulting test_suite.
        srcs: Testdata files (DSLX) to run regressions for.
        prefix: Prefix directory path for the testdata files.
        failing: Optional list of failing testdata paths (must be a string
            match with a member in srcs).
        tags: Optional mapping of testdata paths to additional tags (must
            be a string match with a member in srcs).
    """
    names = []
    tags = tags or {}
    for f in srcs:
        if not f.endswith(".x"):
            fail()

        test_name = "run_crasher_test_{}".format(_to_suffix(f))
        names.append(test_name)
        fullpath = prefix + "/" + f
        native.sh_test(
            name = test_name,
            srcs = ["//xls/dslx/fuzzer:run_crasher_sh"],
            args = [fullpath],
            data = [
                "//xls/dslx/fuzzer:run_crasher",
                f,
            ],
            tags = tags.get(f, []),
        )

    native.test_suite(
        name = name,
        tests = names,
        tags = ["manual"],
    )
