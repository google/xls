// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_COMMENT_DATA_H_
#define XLS_DSLX_FRONTEND_COMMENT_DATA_H_

#include <string>

#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// As we encounter comments in the source text we keep sideband notes about them
// outside of the token stream.
struct CommentData {
  Span span;
  std::string text;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_COMMENT_DATA_H_
