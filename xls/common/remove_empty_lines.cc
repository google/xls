#include "xls/common/remove_empty_lines.h"

#include "absl/strings/str_split.h"
#include "absl/strings/str_join.h"
#include "absl/strings/ascii.h"

namespace xls {

std::string RemoveEmptyLines(std::string_view s) {
  std::vector<std::string_view> lines = absl::StrSplit(s, '\n');
  std::vector<std::string_view> nonempty;
  for (std::string_view line : lines) {
    std::string_view stripped = absl::StripAsciiWhitespace(line);
    if (!stripped.empty()) {
      nonempty.push_back(line);
    }
  }
  return absl::StrJoin(nonempty, "\n");
}

}  // namespace xls
