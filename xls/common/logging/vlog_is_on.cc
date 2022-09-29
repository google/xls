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

#include "xls/common/logging/vlog_is_on.h"

#include "absl/base/internal/spinlock.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xls/common/logging/errno_saver.h"
#include "xls/common/logging/vlog_is_on.inc"

// Construct a logging site from a logging level and epoch.
inline int32_t Site(int level, int epoch) {
  return ((level & 0x0000FFFF) << 16) | (epoch & 0x0000FFFF);
}

namespace xls {
namespace logging_internal {
// Implementation of fnmatch that does not need 0-termination of
// arguments and does not allocate any heap memory.  We only support
// "*" and "?" wildcards.  No support for bracket expressions [...].
// Unlike fnmatch, wildcards may match /.  "May" because they do so in
// the current implementation, but we don't promise not to change it.
// Likewise backslash-escaping is not supported but we don't promise
// not to change that.
// It's not a static function for the unittest.
bool SafeFNMatch(std::string_view pattern, std::string_view str) {
  while (true) {
    if (pattern.empty()) {
      // `pattern` is exhausted; succeed if all of `str` was consumed matching
      // it.
      return str.empty();
    }
    if (str.empty()) {
      // `str` is exhausted; succeed if `pattern` is empty or all '*'s.
      return pattern.find_first_not_of('*') == pattern.npos;
    }
    if (pattern.front() == '*') {
      pattern.remove_prefix(1);
      if (pattern.empty()) return true;
      do {
        if (SafeFNMatch(pattern, str)) return true;
        str.remove_prefix(1);
      } while (!str.empty());
      return false;
    }
    if (pattern.front() == '?' || pattern.front() == str.front()) {
      pattern.remove_prefix(1);
      str.remove_prefix(1);
      continue;
    }
    return false;
  }
}
}  // namespace logging_internal

// List of per-module log levels from FLAGS_vmodule.  Once created
// each element is never deleted/modified except for the vlog_level:
// other threads will read VModuleInfo blobs w/o locks.  We can't use
// an STL struct here as we wouldn't know when it's safe to
// delete/update it: other threads need to use it w/o locks.
struct VModuleInfo {
  std::string module_pattern;
  bool module_is_path;  // i.e. it contains a path separator
  mutable std::atomic<int> vlog_level;
  const VModuleInfo* next;
};

// Pointer to head of the VModuleInfo list.
// It's a map from module pattern to logging level for those module(s).
// Protected by vmodule_loc in the associated .inc file.
static std::atomic<VModuleInfo*> vmodule_list;

// Logging sites initialize their epochs to zero.  We initialize the
// global epoch to 1, so that all logging sites are initially stale.
std::atomic<int32_t> xls::logging_internal::vlog_epoch{1};

// This can be called very early, so we use SpinLock and RAW_VLOG here.
int SetVLOGLevel(std::string_view module_pattern, int log_level) {
  int result = absl::GetFlag(FLAGS_v);
  bool found = false;
  absl::base_internal::SpinLockHolder l(
      &logging_internal::vmodule_lock);  // protect whole read-modify-write
  for (const VModuleInfo* info = vmodule_list.load(std::memory_order_relaxed);
       info != nullptr; info = info->next) {
    if (info->module_pattern == module_pattern) {
      if (!found) {
        result = info->vlog_level.load(std::memory_order_acquire);
        found = true;
      }
      info->vlog_level.store(log_level, std::memory_order_release);
    } else if (!found && xls::logging_internal::SafeFNMatch(
                             info->module_pattern, module_pattern)) {
      result = info->vlog_level.load(std::memory_order_acquire);
      found = true;
    }
  }
  if (!found) {
    VModuleInfo* info = new VModuleInfo;
    info->module_pattern = std::string(module_pattern);
#ifdef _WIN32
    info->module_is_path =
        module_pattern.find_first_of("/\\") != module_pattern.npos;
#else
    info->module_is_path = module_pattern.find('/') != module_pattern.npos;
#endif
    info->vlog_level.store(log_level, std::memory_order_release);
    info->next = vmodule_list.load(std::memory_order_relaxed);
    vmodule_list.store(info, std::memory_order_release);
  }
  // Increment the epoch, marking all cached logging sites as stale.
  ::xls::logging_internal::vlog_epoch.fetch_add(1, std::memory_order_release);

  if (XLS_VLOG_IS_ON(1)) {
    ABSL_RAW_LOG(INFO, "Set VLOG level for \"%.*s\" to %d",
            static_cast<int>(module_pattern.size()), module_pattern.data(),
            log_level);
  }
  return result;
}

namespace logging_internal {

// NOTE: Individual XLS_VLOG statements cache the integer log level pointers.
// NOTE: This function must not allocate memory or require any locks.
int InitVLOG(std::atomic<int32_t>* site, std::string_view full_path) {
  // protect the errno global in case someone writes:
  // XLS_VLOG(..) << "The last error was " << strerror(errno)
  ErrnoSaver errno_saver_;

  // Get basename for file
  std::string_view basename = full_path;
  {
    const size_t sep = basename.rfind('/');
    if (sep != basename.npos) {
      basename.remove_prefix(sep + 1);
#ifdef _WIN32
    } else {
      const size_t sep = basename.rfind('\\');
      if (sep != basename.npos) basename.remove_prefix(sep + 1);
#endif
    }
  }

  std::string_view stem = full_path, stem_basename = basename;
  {
    const size_t sep = stem_basename.find('.');
    if (sep != stem_basename.npos) {
      stem.remove_suffix(stem_basename.size() - sep);
      stem_basename.remove_suffix(stem_basename.size() - sep);
    }
    if (absl::ConsumeSuffix(&stem_basename, "-inl")) {
      stem.remove_suffix(std::string_view("-inl").size());
    }
  }

  // Fetch the global epoch before fetching the log site state.  Fetch
  // the log site state before traversing the module list.  It is
  // important that our view of the global epoch is no newer than our
  // view of the log site, and that the epoch value we store is no
  // newer than the list we traverse or the log site we fetch.
  int32_t global_epoch = GlobalEpoch();
  int32_t old_site = site->load(std::memory_order_acquire);
  int32_t new_site = Site(SiteLevel(kDefaultSite), global_epoch);

  // Find target in list of modules, and set new_site with a
  // module-specific verbosity level, if found.
  const VModuleInfo* info = vmodule_list.load(std::memory_order_acquire);

  // If we find a matching module in the list, we use its
  // vlog_level to control the VLOG at the call site.  Otherwise,
  // the site remains set to its default value.
  while (info != nullptr) {
    if (info->module_is_path) {
      // If there are any slashes in the pattern, try to match the full fname.
      if (xls::logging_internal::SafeFNMatch(info->module_pattern, stem)) {
        new_site = Site(info->vlog_level.load(std::memory_order_acquire),
                        global_epoch);
        break;
      }
    } else if (xls::logging_internal::SafeFNMatch(info->module_pattern,
                                                  stem_basename)) {
      // Otherwise, just match the basename.
      new_site =
          Site(info->vlog_level.load(std::memory_order_acquire), global_epoch);
      break;
    }
    info = info->next;
  }

  // Attempt to store the new log site.  This can race with other
  // threads.  If we lose the race, don't bother retrying.  Our epoch
  // will remain stale, so we will try again on the next iteration.
  site->compare_exchange_strong(old_site, new_site, std::memory_order_release,
                                std::memory_order_relaxed);

  return SiteLevel(new_site);
}

bool VLogEnabledSlow(std::atomic<int32_t>* site, int32_t level,
                     std::string_view file) {
  const int32_t site_copy = site->load(std::memory_order_acquire);
  int32_t site_level = ABSL_PREDICT_TRUE(SiteEpoch(site_copy) == GlobalEpoch())
                           ? SiteLevel(site_copy)
                           : InitVLOG(site, file);

  if (site_level == kUseFlag) {
    // Use global setting instead of per-site setting.
    site_level = absl::GetFlag(FLAGS_v);
  }
  return ABSL_PREDICT_FALSE(site_level >= level);
}

}  // namespace logging_internal
}  // namespace xls
