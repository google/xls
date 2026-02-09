#ifndef XLS_ECO_MCS_GED_IR_PATCH_GEN_H_
#define XLS_ECO_MCS_GED_IR_PATCH_GEN_H_

#include "graph.h"
#include "xls/eco/ir_patch.pb.h"
#include "xls/eco/mcs_ged/ged.h"

xls_eco::IrPatchProto GenerateIrPatchProto(const XLSGraph& original_graph,
                                           const XLSGraph& modified_graph,
                                           const ged::GEDResult& ged_result);

#endif  // XLS_ECO_MCS_GED_IR_PATCH_GEN_H_
