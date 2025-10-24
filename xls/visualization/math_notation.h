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

#ifndef XLS_VISUALIZATION_MATH_NOTATION_H_
#define XLS_VISUALIZATION_MATH_NOTATION_H_

#include <functional>
#include <string>

#include "xls/ir/node.h"

namespace xls {

std::string ToMathNotation(const Node* node);

std::string ToMathNotation(const Node* node,
                           std::function<bool(const Node*)> treat_as_atom);

}  // namespace xls

#endif  // XLS_VISUALIZATION_MATH_NOTATION_H_
