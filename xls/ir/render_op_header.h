#ifndef XLS_IR_RENDER_OP_HEADER_H_
#define XLS_IR_RENDER_OP_HEADER_H_

#include <string>

namespace xls {

// Renders the `enum class Op` definition.
std::string RenderEnumClassOp();

// Renders the entirety of the `op.h` header file.
std::string RenderOpHeader();

}  // namespace xls

#endif  // XLS_IR_RENDER_OP_HEADER_H_
