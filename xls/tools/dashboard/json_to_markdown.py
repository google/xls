#!/usr/bin/env python3
# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, Optional, Sequence

from mdutils.mdutils import MdUtils

DashboardJSONs = Sequence[Dict]
MarkdownContents = Sequence[str]


def json_to_markdown(
  jsons: DashboardJSONs, title: str = "Dashboard", desc: Optional[str] = None
) -> MarkdownContents:
  md = MdUtils(file_name=None, title=title)

  if desc is not None:
    md.new_paragraph(desc)

  sorted_groups = sorted({x["group"] for x in jsons})
  for group in sorted_groups:
    group_jsons = filter(lambda x: x["group"] == group, jsons)
    group_jsons_sorted = sorted(list(group_jsons), key=lambda x: x["name"])

    md.new_header(level=1, title=group)

    for j in group_jsons_sorted:
      md.new_header(level=2, title=j["name"])

      if "desc" in j:
        md.new_paragraph(j["desc"])
        md.new_line()

      if j["type"] == "table":
        rows = len(j["value"])
        columns = len(j["value"][0])
        text = [x for y in j["value"] for x in y]
        md.new_table(columns, rows, text)
      else:
        raise ValueError("Unsupported value type")

      if "label" in j:
        text = f"test: ``{j['label']}``"
        if "log" in j:
          text = md.new_inline_link(link=j["log"], text=text)

        md.new_paragraph(text)

  return md.get_md_text()
