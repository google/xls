#include <string>
#include <string_view>

namespace xls {

// Removes all empty and whitespace-only lines from the given string `s`.
std::string RemoveEmptyLines(std::string_view s);

}  // namespace xls
