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

#ifndef XLS_NETLIST_FUNCTION_EXTRACTOR_H_
#define XLS_NETLIST_FUNCTION_EXTRACTOR_H_


#include "absl/status/statusor.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {
namespace function {

// This program iterates through the blocks contained in a specified Liberty-
// formatted file and collects the input and output pins of a "cell".
// For output pins, the "function" is collected as well - that specifies the
// logical operation of the cell or pin (in the case of multiple output pins).
absl::StatusOr<CellLibraryProto> ExtractFunctions(cell_lib::CharStream* stream);

}  // namespace function
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_FUNCTION_EXTRACTOR_H_
