// Copyright 2020 The XLS Authors
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
#ifndef XLS_DSLX_CPP_TRANSPILER_H_
#define XLS_DSLX_CPP_TRANSPILER_H_

#include <filesystem>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Converts DSLX types contained inside of "module" into their C++ equivalents.
// For enums and constant declarations, this simply dumps the elements into the
// returned text, using the smallest containing type for the latter, e.g., a
// uint16_t for a 13-bit element. Structs use more complicated getters and
// setters and converters into XLS Value types, each with appropriate size and
// completeness validation. See the associated unit test for concrete examples.
//
// Note that the given Module must have been typechecked.
//
// The APIs emitted here are not guaranteed to be stable over time. For example,
// we may define a C++ type "xls::u7" to represent a seven-bit quantity. That
// being said, no such changes are planned (as of this writing) and any changes
// should be infrequent, so users should feel comfortable using these
// interfaces, but should also be aware of the potential for change in the
// future.
//
// TODO(rspringer): 2021-06-09 Currently all outputs are placed in a single
// file, the header, but this will likely be split up once structs are
absl::StatusOr<std::string> TranspileToCpp(
    Module* module, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_TRANSPILER_H_
