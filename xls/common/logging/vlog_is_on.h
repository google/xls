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

#include <atomic>
#include <cstdint>
#include <string>

#include "absl/base/optimization.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"

ABSL_DECLARE_FLAG(int32_t, v);
// Note: Setting vmodule with absl::SetFlag is not supported. Instead use
// xls::SetVLOGLevel.
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
#define XLS_VLOG_IS_ON(verbose_level)               \
  (::xls::logging_internal::VLogEnabled(            \
      []() -> std::atomic<int32_t>* {               \
        static std::atomic<int32_t> site__(         \
            ::xls::logging_internal::kDefaultSite); \
        return &site__;                             \
      }(),                                          \
      (verbose_level), __FILE__))

namespace xls {

// Set XLS_VLOG(_IS_ON) level for module_pattern to log_level.
// This lets us dynamically control what is normally set by the --vmodule flag.
// Returns the level that previously applied to module_pattern.
// NOTE: To change the log level for VLOG(_IS_ON) sites
//       that have already executed after/during InitGoogle,
//       one needs to supply the exact --vmodule pattern that applied to them.
//       (If no --vmodule pattern applied to them
//       the value of FLAGS_v will continue to control them.)
int SetVLOGLevel(std::string_view module_pattern, int log_level);

// Private implementation details.  No user-serviceable parts inside.
namespace logging_internal {

// Each log site determines whether its log level is up to date by
// comparing its epoch to this global epoch.  Whenever the program's
// vmodule configuration changes (ex: SetVLOGLevel is called), the
// global epoch is advanced, invalidating all site epochs.
extern std::atomic<int32_t> vlog_epoch;

// A log level of kUseFlag means "read the logging level from FLAGS_v."
const int kUseFlag = (int16_t)~0x7FFF;

// Log sites use FLAGS_v by default, and have an initial epoch of 0.
const int32_t kDefaultSite = static_cast<unsigned int>(kUseFlag) << 16;

// The global epoch is the least significant half of an int32_t, and
// may only be accessed through atomic operations.
inline int32_t GlobalEpoch() {
  return vlog_epoch.load(std::memory_order_acquire) & 0x0000FFFF;
}

// The least significant half of a site is the epoch.
inline int SiteEpoch(int32_t site) { return site & 0x0000FFFF; }

// The most significant half of a site is the logging level.
inline int SiteLevel(int32_t site) { return site >> 16; }

// Attempt to initialize or reinitialize a VLOG site.  Returns the
// level of the log site, regardless of whether the attempt succeeds
// or fails.
//   site: The address of the log site's state.
//   fname: The filename of the current source file.
int InitVLOG(std::atomic<int32_t>* site, std::string_view full_path);

// Slow path version of VLogEnabled.
bool VLogEnabledSlow(std::atomic<int32_t>* site, int32_t level,
                     std::string_view file);

// Determine whether verbose logging should occur at a given log site. Fast path
// is inlined, slow path is delegated to VLogEnabledSlow.
// This uses ABSL_ATTRIBUTE_ALWAYS_INLINE because callers expect VLOG to have
// minimal overhead, and because GCC doesn't do this unless it is forced.
// Inlining the function yields a ~3x performance improvement at the cost of a
// 1.5x code size increase at the call site.
#if defined(__GNUC__) && !defined(__clang__)
ABSL_ATTRIBUTE_ALWAYS_INLINE
#endif
inline bool VLogEnabled(std::atomic<int32_t>* site, int32_t level,
                        std::string_view file) {
  const int32_t site_copy = site->load(std::memory_order_acquire);
  if (ABSL_PREDICT_TRUE(SiteEpoch(site_copy) == GlobalEpoch())) {
    int32_t site_level = SiteLevel(site_copy);
    if (site_level == kUseFlag) {
      // Use global setting instead of per-site setting.
      site_level = absl::GetFlag(FLAGS_v);
    }
    if (ABSL_PREDICT_TRUE(level > site_level)) {
      return false;
    }
  }
  return VLogEnabledSlow(site, level, file);
}

}  // namespace logging_internal

// Represents a unique callsite for a `XLS_VLOG()` or `XLS_VLOG_IS_ON()` call.
// Helper libraries that provide vlog like functionality should use this to
// efficiently handle -vmodule.
//
// Notionally the site also includes a `std::string_view` file, but putting
// that in the class would increase the size, so the consistency invariant is
// pushed onto callers of `IsEnabled()`.
class VLogSite {
 public:
  constexpr VLogSite() {}

  // Since this is a caching location, copying it doesn't make sense.  The copy
  // should get a brand new kDefaultSite.
  VLogSite(const VLogSite&) = delete;
  VLogSite& operator=(const VLogSite&) = delete;
  VLogSite(VLogSite&&) = delete;
  VLogSite& operator=(VLogSite&&) = delete;

  // Returns true if logging is enabled.  Like XLS_VLOG_IS_ON(2), this uses
  // internal atomics to be fast in the common case.  For any given instance of
  // `VLogSite`, all calls to `IsEnabled` must use the same value for `file`.
  bool IsEnabled(int32_t level, std::string_view file) const {
    return logging_internal::VLogEnabled(&site_, level, file);
  }

 private:
  mutable std::atomic<int32_t> site_{logging_internal::kDefaultSite};
};

}  // namespace xls

namespace base_logging {
namespace logging_internal {
bool SafeFNMatch(std::string_view pattern, std::string_view str);
}  // namespace logging_internal
}  // namespace base_logging

#endif  // XLS_COMMON_LOGGING_VLOG_IS_ON_H_
