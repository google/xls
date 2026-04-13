// Copyright 2025 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_ECO_GXL_PARSER_H_
#define XLS_ECO_GXL_PARSER_H_

#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/libs/tinyxml2/tinyxml2.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

XLSGraph parse_gxl(const std::string &filename);
bool export_gxl(const XLSGraph &graph, const std::string &filename);

#endif  // XLS_ECO_GXL_PARSER_H_
