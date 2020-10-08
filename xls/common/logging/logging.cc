// Copyright 2020 The XLS Authors
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

#include "xls/common/logging/logging.h"

#include "absl/base/attributes.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace xls {
namespace logging_internal {

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << exprtext << " (";
}

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_.get();
}

std::string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new std::string(stream_->str());
}

void MakeCheckOpValueString(std::ostream* os, const char v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream* os, const signed char v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream* os, const unsigned char v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream* os, const void* p) {
  if (p == nullptr) {
    (*os) << "(null)";
  } else {
    (*os) << p;
  }
}

void DieBecauseNull(const char* file, int line, const char* exprtext) {
  XLS_LOG(FATAL)
      .AtLocation(file, line)
      .WithCheckFailureMessage(
          absl::StrCat("'", exprtext, "' Must be non-null"));
}

}  // namespace logging_internal
}  // namespace xls
