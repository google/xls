#ifndef XLS_IR_RENDER_NODES_HEADER_H_
#define XLS_IR_RENDER_NODES_HEADER_H_

#include <string>

#include "xls/ir/op_specification.h"

namespace xls {

// Renders the class declaration for the given `op_class`.
//
// Exposed for unit testing.
std::string RenderNodeSubclass(const OpClass& op_class);

// Renders the entirety of the `nodes.h` header file.
std::string RenderNodesHeader();

}  // namespace xls

#endif  // XLS_IR_RENDER_NODES_HEADER_H_
