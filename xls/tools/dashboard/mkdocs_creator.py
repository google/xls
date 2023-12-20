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


import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Union

from mkdocs.commands.build import build as mkdocs_build
from mkdocs.commands.new import new as mkdocs_new
from mkdocs.config.defaults import MkDocsConfig

default_config = {
  "site_name": "Dashboard",
}

CssPath = str
CssContents = Union[str, Sequence[str]]
MarkdownPath = str
MarkdownContents = Union[str, Sequence[str]]
SystemPath = Union[str, Path]


class MkDocsCreator:
  def __init__(self, config: Dict = default_config):
    self.config: Dict = config
    self.contents: Dict[MarkdownPath, MarkdownContents] = {}
    self.css: Dict[CssPath, CssContents] = {}

  def _docs_path(self, outputdir: SystemPath) -> Path:
    """Given the output directory, returns the default directory for markdown files"""
    return Path(outputdir, "docs")

  def _html_path(self, outputdir: SystemPath) -> Path:
    """Given the output directory, returns the default directory for generated HTML files"""
    return Path(outputdir, "site")

  def add_page(self, path: MarkdownPath, contents: MarkdownContents) -> None:
    """Adds new markdown page"""
    if path in self.contents:
      raise ValueError("Markdown file already exists")
    self.contents.update({path: contents})

  def add_index(self, contents: MarkdownContents) -> None:
    """Adds new markdown index page"""
    self.add_page("index", contents)

  def add_css(self, contents: CssContents, name: str = "extra.css") -> None:
    if name in self.css:
      raise ValueError("CSS file already exists")
    self.css.update({name: contents})

    if "extra_css" not in self.config:
      self.config["extra_css"] = []

    self.config["extra_css"].append(name)

  def generate(self, outputdir: SystemPath) -> None:
    """Generates MkDocs sources of the website"""
    output_path = Path(outputdir)
    mkdocs_new(str(output_path))

    docsdir = self._docs_path(output_path)
    for path, contents in self.contents.items():
      page_path = docsdir / f"{path}.md"

      if not page_path.parent.is_dir():
        os.makedirs(page_path.parent)

      with open(page_path, "w") as fp:
        if isinstance(contents, str):
          fp.write(contents)
        else:
          for line in contents:
            fp.write(f"{line}\n")

    for css, contents in self.css.items():
      page_path = docsdir / css
      with open(page_path, "w") as fp:
        if isinstance(contents, str):
          fp.write(contents)
        else:
          for line in contents:
            fp.write(f"{line}\n")

  def build(self, outputdir: SystemPath) -> None:
    """Generates HTML sources of the website"""
    output_path = Path(outputdir)
    if output_path.is_dir():
      raise FileExistsError(f"{output_path} output path already exists!")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as fd:
      os.chdir(fd)
      cfg = MkDocsConfig()
      try:
        self.generate(fd)
        cfg.load_dict(self.config)
        cfg.validate()
        cfg.plugins.on_startup(command="build", dirty=False)
        mkdocs_build(cfg)
      finally:
        cfg.plugins.on_shutdown()
        os.chdir(cwd)
      shutil.move(self._html_path(fd), output_path)


class DefaultDashboardCreator(MkDocsCreator):
  EXTRA_CSS = """
  .md-typeset__table {
    min-width: 100%;
  }

  .md-typeset table:not([class]) {
    display: table;
  }

  .md-typeset table:not([class]) td:not(:first-child) {
    border-left: .05rem solid var(--md-typeset-table-color);
  }

  .md-nav {
    visibility: hidden;
  }
  """

  def __init__(self, name, page):
    config = {
      "site_name": name,
      "theme": {
        "name": "material",
      },
      "use_directory_urls": False,
    }

    super().__init__(config)
    self.add_index(page)
    self.add_css(self.EXTRA_CSS)
