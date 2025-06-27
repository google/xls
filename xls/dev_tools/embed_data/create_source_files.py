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


"""Create source and header files for embedding binary data in a cc library."""

from collections.abc import Sequence
import dataclasses

from absl import app
from absl import flags
import jinja2

from xls.common import runfiles


_NAMESPACE = flags.DEFINE_string(
    "namespace",
    default=None,
    required=True,
    help="Namespace to place the data in.",
)
_ACCESSOR = flags.DEFINE_string(
    "accessor",
    default=None,
    required=True,
    help="function to access the data from.",
)
_DATA_FILE = flags.DEFINE_string(
    "data_file",
    default=None,
    required=True,
    help="File containing the data to be embedded.",
)
_OUTPUT_SOURCE = flags.DEFINE_string(
    "output_source",
    default=None,
    required=True,
    help="Path at which to write the output source file.",
)
_OUTPUT_HEADER = flags.DEFINE_string(
    "output_header",
    default=None,
    required=True,
    help="Path at which to write the output header file.",
)


@dataclasses.dataclass
class Embedding:
  namespace: str
  accessor: str
  data_file: str
  header_file: str
  data_file_contents: str


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with open(_DATA_FILE.value, "rb") as data:
    contents = data.read()
  emb = Embedding(
      namespace=_NAMESPACE.value,
      accessor=_ACCESSOR.value,
      data_file=_DATA_FILE.value,
      header_file=_OUTPUT_HEADER.value,
      data_file_contents=", ".join(str(int(x)) for x in contents),
  )
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  with open(_OUTPUT_SOURCE.value, "wt") as cc_file:
    cc_template = env.from_string(
        runfiles.get_contents_as_text(
            "xls/dev_tools/embed_data/embedded_data.cc.tmpl"
        )
    )
    cc_file.write("// Generated File. Do not edit.\n")
    cc_file.write(cc_template.render({"embed": emb}))
    cc_file.write("\n")
  with open(_OUTPUT_HEADER.value, "wt") as h_file:
    h_template = env.from_string(
        runfiles.get_contents_as_text(
            "xls/dev_tools/embed_data/embedded_data.h.tmpl"
        )
    )
    h_file.write("// Generated File. Do not edit.\n")
    h_file.write(h_template.render({"embed": emb}))
    h_file.write("\n")


if __name__ == "__main__":
  app.run(main)
