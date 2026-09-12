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

#ifndef XLS_ECO_IR_PATCH_GEN_H_
#define XLS_ECO_IR_PATCH_GEN_H_

#include "xls/contrib/eco/ged.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/ir_patch.pb.h"

xls_eco::IrPatchProto GenerateIrPatchProto(const XLSGraph& original_graph,
                                           const XLSGraph& modified_graph,
                                           const ged::GEDResult& ged_result);

#endif  // XLS_ECO_IR_PATCH_GEN_H_
