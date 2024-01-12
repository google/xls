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

#ifndef XLS_IR_PROC_CONVERSION_H_
#define XLS_IR_PROC_CONVERSION_H_

#include "absl/status/status.h"
#include "xls/ir/package.h"

namespace xls {

// Convert a package containing old-style procs to a package of new-style
// procs. Top must be set. The hierarchy is constructed according to the
// following rules:
//
//   (0) All non-top procs are instantiated by the top proc in a flat hierarchy.
//
//   (1) All non-kSendReceive channels are declared as interface channels in the
//       top proc.
//
//   (2) All kSendReceive channels are declared in the top proc scope unless the
//       channel is a loopback channel in which case the channel is declared in
//       the proc in which it is used.
//
// All global channels are removed from the package after the hierarchy is
// constructed.
absl::Status ConvertPackageToNewStyleProcs(Package* package);

}  // namespace xls

#endif  // XLS_IR_PROC_CONVERSION_H_
