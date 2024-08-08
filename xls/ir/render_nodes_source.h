#ifndef XLS_IR_RENDER_NODES_SOURCE_H_
#define XLS_IR_RENDER_NODES_SOURCE_H_

#include <string>

#include "xls/ir/op_specification.h"

namespace xls {

// Renders the constructor for the given `op_class`.
//
// Exposed for unit testing.
std::string RenderConstructor(const OpClass& op_class);

// Renders the creation of a standard clone method for the given `op_class`.
//
// Exposed for unit testing.
std::string RenderStandardCloneMethod(const OpClass& op_class);

// Renders the entirety of the `nodes.cc` source file.
std::string RenderNodesSource();

}  // namespace xls

#endif  // XLS_IR_RENDER_NODES_SOURCE_H_
