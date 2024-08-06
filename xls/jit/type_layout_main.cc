// Copyright 2024 The XLS Authors
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

#include <unistd.h>

#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/jit/type_layout.h"
#include "xls/jit/type_layout.pb.h"

const char kUsage[] = R"(
Flatten a 'value' into a jit-compatible byte array (written to the stdout).
)";

ABSL_FLAG(std::optional<std::string>, layout_proto, std::nullopt,
          "File containing a (binary) TypeLayout proto");
ABSL_FLAG(std::optional<std::string>, layout_textproto, std::nullopt,
          "File containing a textproto TypeLayout proto");
ABSL_FLAG(std::optional<std::string>, encode, std::nullopt,
          "The actual Value to turn into a binary buffer in the standard xls "
          "value format");
ABSL_FLAG(bool, decode, false,
          "If set, read bytes from stdin and print the value representation to "
          "stdout.");
ABSL_FLAG(bool, mask, false,
          "If set, write a mask where bits which are not part of the actual "
          "value are 0.");
namespace xls {
namespace {

absl::StatusOr<std::string> Encode(const TypeLayout& layout,
                                   std::string_view value) {
  XLS_ASSIGN_OR_RETURN(Value ir_val, Parser::ParseTypedValue(value),
                       _ << "Invalid value");
  std::string buf;
  buf.resize(layout.size());
  layout.ValueToNativeLayout(ir_val, reinterpret_cast<uint8_t*>(buf.data()));
  return buf;
}
absl::StatusOr<std::string> Decode(const TypeLayout& layout,
                                   std::string_view data) {
  XLS_RET_CHECK_EQ(data.size(), layout.size());
  return layout
      .NativeLayoutToValue(reinterpret_cast<const uint8_t*>(data.data()))
      .ToString();
}

absl::Status RealMain(const std::optional<std::string>& layout_proto,
                      const std::optional<std::string>& layout_textproto,
                      const std::optional<std::string>& value, bool decode,
                      bool mask) {
  if (layout_textproto.has_value() == layout_proto.has_value()) {
    return absl::InvalidArgumentError(
        "Exactly one of --layout_proto or --layout_textproto must be set.");
  }
  TypeLayoutProto proto;
  if (layout_proto) {
    XLS_ASSIGN_OR_RETURN(std::string proto_bytes,
                         GetFileContents(*layout_proto));
    XLS_RET_CHECK(proto.ParseFromString(proto_bytes))
        << "Unable to parse type data";
  } else {
    XLS_ASSIGN_OR_RETURN(std::string proto_bytes,
                         GetFileContents(*layout_textproto));
    XLS_RETURN_IF_ERROR(ParseTextProto(proto_bytes, *layout_textproto, &proto));
  }
  if (value && decode) {
    return absl::InvalidArgumentError(
        "Only one of --decode and --encode may be set.");
  }
  Package pkg("temp_package");
  XLS_ASSIGN_OR_RETURN(TypeLayout layout, TypeLayout::FromProto(proto, &pkg),
                       _ << "Invalid layout");
  std::string out;
  if (value) {
    XLS_ASSIGN_OR_RETURN(out, Encode(layout, *value));
  } else if (mask) {
    std::vector<uint8_t> res = layout.mask();
    out.resize(res.size(), '\0');
    absl::c_transform(res, out.begin(),
                      [](uint8_t v) -> char { return std::bit_cast<char>(v); });
  } else {
    XLS_ASSIGN_OR_RETURN(std::string raw, GetFileContents("/dev/stdin"));
    XLS_ASSIGN_OR_RETURN(out, Decode(layout, raw));
  }
  // TODO(allight): Technically not posix since write can write only part of the
  // data or be interrupted but generally that won't happen with our hosts.
  write(STDOUT_FILENO, out.data(), out.size());
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Expected invocation: " << argv[0];
  }

  return xls::ExitStatus(xls::RealMain(
      absl::GetFlag(FLAGS_layout_proto), absl::GetFlag(FLAGS_layout_textproto),
      absl::GetFlag(FLAGS_encode), absl::GetFlag(FLAGS_decode),
      absl::GetFlag(FLAGS_mask)));
}
