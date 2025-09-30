#
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

"""Emits a shell script that runs VVP on a genrule-created iverilog-output.

We write this as a Python script so it's slightly less mind-boggling than bash
programs emitting bash, and so we have nice string interpolation.
"""

from absl import app


def main(argv):
  if len(argv) != 2:
    raise app.UsageError(
        'Bad number of command-line arguments; expect '
        '<generate_vvp_runner> <path_of_iverilog_output>'
    )

  path = argv[1]
  # Strip off the leading portion of the path that's used in genrules; i.e. the
  # first three elements of:
  #
  # bazel-out/k8-fastbuild/bin/xls/[...]
  path = '/'.join(path.split('/')[3:])

  print("""#!/usr/bin/env bash
set -e
temp=$(mktemp)
stdout="${{temp}}.stdout"
stderr="${{temp}}.stderr"
file={path!r}
./third_party/iverilog/vvp -M ./third_party/iverilog $file > $stdout 2> $stderr
cat $stdout
cat $stderr >&2
errors=$(cat $stdout $stderr | grep -c -i error)
exit $errors
""".format(path=path))


if __name__ == '__main__':
  app.run(main)
