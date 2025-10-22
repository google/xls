// Copyright 2025 The XLS Authors
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

#ifndef XLS_TOOLS_DELAY_INFO_FLAGS_H_
#define XLS_TOOLS_DELAY_INFO_FLAGS_H_

#include <string_view>

#include "xls/tools/delay_info_flags.pb.h"

namespace xls {

// Returns the delay info flags from the command line, specifying the given
// `input_path` in the returned proto, since it is not a named command line
// flag.
DelayInfoFlagsProto GetDelayInfoFlagsProto(std::string_view input_path);

}  // namespace xls

#endif  // XLS_TOOLS_DELAY_INFO_FLAGS_H_
