#ifndef XLS_ECO_IR_PATCH_GEN_H_
#define XLS_ECO_IR_PATCH_GEN_H_

#include "xls/eco/ged.h"
#include "xls/eco/graph.h"
#include "xls/eco/ir_patch.pb.h"

xls_eco::IrPatchProto GenerateIrPatchProto(const XLSGraph& original_graph,
                                           const XLSGraph& modified_graph,
                                           const ged::GEDResult& ged_result);

#endif  // XLS_ECO_IR_PATCH_GEN_H_
