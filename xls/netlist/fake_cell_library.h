// Copyright 2020 Google LLC
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

#ifndef XLS_NETLIST_FAKE_CELL_LIBRARY_H_
#define XLS_NETLIST_FAKE_CELL_LIBRARY_H_

#include "xls/netlist/netlist.h"

namespace xls {
namespace netlist {

// Creates a fake cell library suitable for testing.
CellLibrary MakeFakeCellLibrary();

}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_FAKE_CELL_LIBRARY_H_
