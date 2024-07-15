#ifndef XLS_IR_RENDER_NODES_HEADER_H_
#define XLS_IR_RENDER_NODES_HEADER_H_

#include <string>

#include "xls/ir/op_specification.h"

namespace xls {

std::string RenderNodeSubclass(const OpClass& op_class);

std::string RenderNodesHeader();

}  // namespace xls

#endif  // XLS_IR_RENDER_NODES_HEADER_H_
