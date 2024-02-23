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

// Defines the XLS_VLOG_IS_ON macro that controls the variable-verbosity
// conditional logging.
//
// It's used by XLS_VLOG and XLS_VLOG_IF in logging.h to trigger the logging.
//
// It can also be used directly e.g. like this:
//   if (XLS_VLOG_IS_ON(2)) {
//     // do some logging preparation and logging
//     // that can't be accomplished e.g. via just XLS_VLOG(2) << ...;
//   }
//
// The truth value that XLS_VLOG_IS_ON(level) returns is determined by
// the two verbosity level flags:
//   --v=<n>  Gives the default maximal active V-logging level;
//            0 is the default.
//            Normally positive values are used for V-logging levels.
//   --vmodule=<str>  Gives the per-module maximal V-logging levels to override
//                    the value given by --v.  If the pattern contains a slash,
//                    the full filename is matched against the pattern;
//                    otherwise only the basename is used.
//                    E.g. "my_module=2,foo*=3,*/bar/*=4" would change
//                    the logging level for all code in source files
//                    "my_module.*", "foo*.*", and all files with directory
//                    "bar" in their path ("-inl" suffixes are also disregarded
//                    for this matching).

#ifndef XLS_COMMON_LOGGING_VLOG_IS_ON_H_
#define XLS_COMMON_LOGGING_VLOG_IS_ON_H_

#include <cstdint>
#include <string>

#include "absl/flags/declare.h"
#include "absl/log/log.h"

ABSL_DECLARE_FLAG(int32_t, v);
// Note: Setting vmodule with absl::SetFlag is not supported. Instead use
// absl::SetVLogLevel.
ABSL_DECLARE_FLAG(std::string, vmodule);

// We pack an int16_t verbosity level and an int16_t epoch into an
// int32_t at every XLS_VLOG_IS_ON() call site.  The level determines
// whether the site should log, and the epoch determines whether the
// site is stale and should be reinitialized.  A verbosity level of
// kUseFlag (kint16min) indicates that the value of FLAGS_v should be used as
// the verbosity level.  When the site is (re)initialized, a verbosity
// level for the current source file is retrieved from an internal
// list.  This list is mutated through calls to SetVLOGLevel() and
// mutations to the --vmodule flag.  New log sites are initialized
// with a stale epoch and a verbosity level of kUseFlag.
#define XLS_VLOG_IS_ON(verbose_level) VLOG_IS_ON(verbose_level)


#endif  // XLS_COMMON_LOGGING_VLOG_IS_ON_H_
