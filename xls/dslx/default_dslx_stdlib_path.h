// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_DEFAULT_DSLX_STDLIB_PATH_H_
#define XLS_DSLX_DEFAULT_DSLX_STDLIB_PATH_H_

#include <string_view>
#include <type_traits>

#include "xls/dslx/default_dslx_stdlib_path.inc"

// Check that the included file defines ::xls::kDefaultDslxStdlibPath as a
// std::string_view.
static_assert(std::is_same_v<
              std::string_view,
              std::remove_cvref_t<decltype(::xls::kDefaultDslxStdlibPath)>>);

#endif  // XLS_DSLX_DEFAULT_DSLX_STDLIB_PATH_H_
