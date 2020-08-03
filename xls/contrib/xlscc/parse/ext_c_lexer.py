# Lint as: python3
#
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

# pylint: disable=C6102,C6108

"""Extends pycparser's lexing with select C++ features."""

from pycparser.c_lexer import CLexer as CLexerBase


class XLSccLexer(CLexerBase):
  """Extends CLexer with select C++ features.
  """
  tokens = CLexerBase.tokens + ('DBLCOLON',)

  t_DBLCOLON = r'::'

def add_lexer_keywords(cls, keywords):
  cls.keywords = cls.keywords + tuple(kw.upper() for kw in keywords)

  cls.keyword_map = cls.keyword_map.copy()
  cls.keyword_map.update(dict((kw, kw.upper()) for kw in keywords))

  cls.tokens = cls.tokens + tuple(kw.upper() for kw in keywords)


_CL_KEYWORDS = ['BOOL', 'TRUE', 'FALSE', 'TEMPLATE', 'CLASS']
add_lexer_keywords(XLSccLexer, [str.lower(kw) for kw in _CL_KEYWORDS])
