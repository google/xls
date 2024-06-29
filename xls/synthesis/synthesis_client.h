// Copyright 2023 The XLS Authors
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

#ifndef XLS_SYNTHESIS_SYNTHESIS_CLIENT_H_
#define XLS_SYNTHESIS_SYNTHESIS_CLIENT_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/synthesis/synthesis.pb.h"

namespace xls {
namespace synthesis {

// This creates a new channel and stub *each* invocation
absl::StatusOr<CompileResponse> SynthesizeViaClient(
    const std::string& server, const CompileRequest& request);

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_SYNTHESIS_SYNTHESIS_CLIENT_H_
