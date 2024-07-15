#ifndef XLS_IR_RENDER_NODES_SOURCE_H_
#define XLS_IR_RENDER_NODES_SOURCE_H_

#include <string>

#include "xls/ir/op_specification.h"

namespace xls {

std::string RenderConstructor(const OpClass& op_class);

std::string RenderStandardCloneMethod(const OpClass& op_class);

std::string RenderNodesSource();

}  // namespace xls

#endif  // XLS_IR_RENDER_NODES_SOURCE_H_
