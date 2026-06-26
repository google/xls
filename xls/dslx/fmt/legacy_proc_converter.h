// Copyright 2026 The XLS Authors
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

#ifndef XLS_DSLX_FMT_LEGACY_PROC_CONVERTER_H_
#define XLS_DSLX_FMT_LEGACY_PROC_CONVERTER_H_

#include <memory>

#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"

namespace xls::dslx {

// Creates a `Formatter` that reformats legacy procs into impl-style procs, in
// addition to general formatting.
std::unique_ptr<Formatter> CreateLegacyProcConverter(Comments& comments,
                                                     DocArena& arena);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FMT_LEGACY_PROC_CONVERTER_H_
