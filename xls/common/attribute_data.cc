// Copyright 2026 The XLS Authors
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

#include "xls/common/attribute_data.h"

#include <string>

namespace xls {

std::string AttributeKindToString(AttributeKind kind) {
  switch (kind) {
    case AttributeKind::kCfg:
      return "cfg";
    case AttributeKind::kDerive:
      return "derive";
    case AttributeKind::kDslxFormatDisable:
      return "dslx_format_disable";
    case AttributeKind::kExternVerilog:
      return "extern_verilog";
    case AttributeKind::kSvType:
      return "sv_type";
    case AttributeKind::kTest:
      return "test";
    case AttributeKind::kTestProc:
      return "test_proc";
    case AttributeKind::kQuickcheck:
      return "quickcheck";
    case AttributeKind::kChannelStrictness:
      return "channel_strictness";
  }
}

}  // namespace xls
