// Copyright 2022 The XLS Authors
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

// Protected API header that exposes the XLS netlist APIs; e.g. Netlist,
// CellLibrary, netlist scanner.
//
// These are in the "protected" directory as they are internal APIs that are
// allow-listed to particular projects -- those projects accept that these
// libraries are generally implementation details and so are subject to change.

#ifndef XLS_PROTECTED_NETLIST_H_
#define XLS_PROTECTED_NETLIST_H_

// IWYU pragma: begin_exports
#include "xls/netlist/cell_library.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/interpreter.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"
// IWYU pragma: end_exports

#endif  // XLS_PROTECTED_NETLIST_H_
