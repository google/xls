# Copyright 2025 The XLS Authors
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

"""Create a standard library which registers a pass to the registry."""

from collections.abc import Sequence
import dataclasses
from absl import app
from absl import flags
import jinja2
from xls.common import runfiles

_HEADERS = flags.DEFINE_multi_string(
    "header_files",
    required=False,
    help="Header files which export the pass",
    default=[],
)

_OUTPUT_SOURCE = flags.DEFINE_string(
    "output_source",
    default=None,
    required=True,
    help="Path at which to write the output source file.",
)
_PASS_CLASS = flags.DEFINE_string(
    "pass_class",
    required=True,
    help="Name of the class to register.",
    default=None,
)
_SHORT_NAME = flags.DEFINE_string(
    "short_name",
    required=False,
    help="Short name of the pass.",
    default=None,
)
_REGISTRATION_NAME = flags.DEFINE_string(
    "registration_name",
    required=True,
    default=None,
    help="Unique string to give the module initialization.",
)


@dataclasses.dataclass(frozen=True)
class Registration:
  """Information for generating the registration."""

  headers: Sequence[str]
  registration_name: str
  name: str
  short_name: str


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  pass_info = Registration(
      headers=_HEADERS.value,
      registration_name=_REGISTRATION_NAME.value,
      name=_PASS_CLASS.value,
      short_name=f'"{_SHORT_NAME.value}"'
      if _SHORT_NAME.present
      else f"{_PASS_CLASS.value}::kName",
  )
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  with open(_OUTPUT_SOURCE.value, "wt") as cc_file:
    cc_template = env.from_string(
        runfiles.get_contents_as_text(
            "xls/passes/tools/pass_registration.cc.tmpl"
        )
    )
    cc_file.write("// Generated File. Do not edit.\n")
    cc_file.write(cc_template.render({"reg": pass_info}))
    cc_file.write("\n")


if __name__ == "__main__":
  app.run(main)
