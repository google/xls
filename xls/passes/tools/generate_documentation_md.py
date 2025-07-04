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

"""Generates markdown documentation page for passes."""

from collections.abc import Sequence
import dataclasses

from absl import app
from absl import flags
import jinja2

from xls.common import runfiles
from xls.passes import optimization_pass_pipeline_pb2
from xls.passes.tools import pass_documentation_pb2

# TODO(allight): This should be configurable.
_PIPELINE_TXTPB = "/xls/passes/optimization_pass_pipeline.txtpb"
_PASSES = flags.DEFINE_multi_string(
    name="passes",
    default=None,
    required=True,
    help="All the individual documentation protos for each pass library.",
)
_PIPELINE_PROTO = flags.DEFINE_string(
    name="pipeline",
    default=None,
    required=True,
    help="OptimizationPassPipelineProto for the default pipeline.",
)
_LINK_FORMAT = flags.DEFINE_string(
    "link_format",
    required=True,
    default=None,
    help="format to use for links to src",
)
_OUTPUT = flags.DEFINE_string(
    "output", default=None, required=True, help="result markdown file."
)


@dataclasses.dataclass
class PassInfo:
  short_name: str
  long_name: str
  header_link_text: str
  header_link: str
  has_min_opt_level: bool
  min_opt_level: int
  has_max_opt_level: bool
  max_opt_level: int
  compound: Sequence[str] | None
  fixedpoint: bool
  description: str


def _header_linkify(name):
  return "".join(n for n in name if n.isalnum() or n == "_" or n == "-")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with open(_PIPELINE_PROTO.value, "rb") as pipeline_file:
    pipeline = (
        optimization_pass_pipeline_pb2.OptimizationPipelineProto.FromString(
            pipeline_file.read()
        )
    )
  protos = []
  for f in _PASSES.value:
    protos.extend(
        pass_documentation_pb2.PassDocumentationProto.FromString(
            open(f, "rb").read()
        ).passes
    )
  infos = []
  for p in protos:
    infos.append(
        PassInfo(
            short_name=p.short_name,
            long_name=p.long_name,
            header_link_text="Header",
            # TODO(allight): We could probably collect the line numbers easily
            # enough to make this link better.
            header_link=_LINK_FORMAT.value % p.file,
            has_min_opt_level=False,
            min_opt_level=0,
            has_max_opt_level=False,
            max_opt_level=0,
            compound=None,
            fixedpoint=False,
            description=p.notes.strip(),
        )
    )
  for comp in pipeline.compound_passes:
    infos.append(
        PassInfo(
            short_name=comp.short_name,
            long_name=comp.long_name,
            header_link_text="Text-proto",
            # TODO(allight): We could probably collect the line numbers easily
            # enough to make this link better.
            header_link=_LINK_FORMAT.value % _PIPELINE_TXTPB,
            has_min_opt_level=comp.options.HasField("min_opt_level"),
            min_opt_level=comp.options.min_opt_level,
            has_max_opt_level=comp.options.HasField("cap_opt_level"),
            max_opt_level=comp.options.cap_opt_level,
            compound=comp.passes,
            fixedpoint=comp.fixedpoint,
            description=comp.comment.strip(),
        )
    )
  infos.sort(key=lambda v: v.short_name)
  infos.insert(
      0,
      PassInfo(
          short_name="default_pipeline",
          long_name="The default pipeline.",
          header_link_text="Text-proto",
          # TODO(allight): We could probably collect the line numbers easily
          # enough to make this link better.
          header_link=_LINK_FORMAT.value % _PIPELINE_TXTPB,
          has_min_opt_level=False,
          min_opt_level=0,
          has_max_opt_level=False,
          max_opt_level=0,
          compound=pipeline.default_pipeline,
          fixedpoint=False,
          description="",
      ),
  )
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  bindings = {
      "passes": infos,
      "header_linkify": _header_linkify,
  }
  with open(_OUTPUT.value, "wt") as md_file:
    md_file.write("<!-- Generated file. Do not edit. -->\n")
    md_file.write(
        "<!-- To regenerate build `xls/passes/tools:rebuild_documentation`"
        " -->\n"
    )
    md_file.write(
        env.from_string(
            runfiles.get_contents_as_text("xls/passes/tools/passes.md.tmpl")
        ).render(bindings)
    )


if __name__ == "__main__":
  app.run(main)
