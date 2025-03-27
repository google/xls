#
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tool to package XLS bazel builds for pre-built XLS releases.

This tool parses bazel manifest files for a set of bazel targets to create a
directory that can be copied and used independently of the bazel cache.  It
is intended to be used as part of a system that will version, build,
and upload pre-built XLS releases.
"""

import os
import shutil
from typing import Optional

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, 'Path to create package under')
flags.DEFINE_multi_string('inc_target', None, 'Include bazel target in package')
flags.DEFINE_string('bazel_bin', './bazel-bin', 'Root directory of bazel-bin')
flags.DEFINE_string(
    'bazel_execroot', './bazel-xls-export', 'Root directory of bazel_execroot'
)

flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('inc_target')


_IGNORED_MANIFEST_FILES = frozenset([
    # This maps to a different .cache/bazel file for each binary.
    # https://github.com/bazelbuild/proposals/blob/main/designs/2022-07-21-locating-runfiles-with-bzlmod.md
    '_repo_mapping',
])


class GlobalManifest:
  """Stores all manifest mappings.

  When packaging multiple targets, each target's manifest will contain
  a mapping of the target dependencies to the actual cached location.  This
  class records the mapping across multiple manifests and ensures that
  a given file maps uniquly to a given cached location.
  """

  def __init__(self):
    self._mapped_files = dict()
    self._orig_manifest = dict()

  def add_mapping(self, f: str, mapping: str, manifest: str):
    """Records or checks association of file with its mapping.

    Args:
      f: File name.
      mapping: Path to the bazel cached file as recorded in the manifest.
      manifest: Path to manifest recording the f->mapping map
    """
    if f in self._mapped_files:
      f_map = self._mapped_files[f]
      orig_manifest = self._orig_manifest[f]

      if f_map != mapping:
        logging.fatal(
            'FATAL: file %s mapped to %s in manifest %s '
            'but originally %s in manifest %s',
            f,
            mapping,
            manifest,
            f_map,
            orig_manifest,
        )
    else:
      self._mapped_files[f] = mapping
      self._orig_manifest[f] = manifest


class Manifest:
  """Stores a single manifest's mapping.

  For each bazel target, a manifest file is created that maps
  the files used by that target to the bazel cached file.
  """

  def __init__(self, global_manifest: GlobalManifest):
    self._global_manifest = global_manifest
    self._all_files = list()
    self._mapped_files = dict()

  @property
  def all_files(self):
    return self._all_files

  def is_unique(self, f: str) -> bool:
    return self._mapped_files[f] is None

  def is_mapped(self, f: str) -> bool:
    return self._mapped_files[f] is not None

  def get_mapping(self, f: str):
    return self._mapped_files[f]

  def read(self, manifest: str):
    """Reads in a single manifest file and records mapping.

    Args:
      manifest: path to manifest file.
    """
    with open(manifest) as f:
      for l in f:
        file_map = l.rstrip().split(maxsplit=1)

        file_name = file_map[0]

        if file_name in _IGNORED_MANIFEST_FILES:
          continue

        if len(file_map) == 1:
          self._all_files.append(file_name)
          self._mapped_files[file_name] = None

          logging.debug(' - %s - copy unique', file_name)
          continue

        file_mapped_name = file_map[1]

        self._all_files.append(file_name)
        self._mapped_files[file_name] = file_mapped_name

        self._global_manifest.add_mapping(file_name, file_mapped_name, manifest)

        logging.debug(' - %s - copy mapped - %s', file_name, file_mapped_name)


class BazelTargetPackager:
  """Copies pre-built bazel targets into a given directory (a package).

  The resulting directory structure is a simplified copy of the directory
  structure created by bazel.  Each target and its runfiles will be copied.
  to the specified output directory.  Runfiles which are symlinks and
  point to the bazel .cache directory will be redirected to
  a files copied under <output_dir>/.common_runfiles.
  """

  def __init__(self, output_dir: str, bazel_bin: str, bazel_execroot: str):
    """Initializes packager.

    Args:
      output_dir: Directory to create package under (should not exist).
      bazel_bin: Path of bazel-bin symlink
      bazel_execroot: Path to bazel-execroot symlink (generally bazel-<proj>).
    """
    # Stores mapping from bazel cached files to actual file under
    # .common_runfiles.  This is used to ensure that even though
    # multiple manifests could refer to the same file, that actual file
    # is copied once and symlinked multiple times.
    self._linked_files = dict()
    self._global_manifest = GlobalManifest()

    # Resolve bazel-bin and bazel-execroot symlinks
    self._bazel_bin = os.path.realpath(bazel_bin)
    self._bazel_execroot = os.path.realpath(bazel_execroot)

    self._bazel_outputroot = os.path.normpath(
        os.path.join(self._bazel_execroot, '../../')
    )

    # Derive .common_runfiles directory from output_dir
    self._output_dir = output_dir
    self._common_runfiles = os.path.join(self._output_dir, '.common_runfiles')

  def create_output_dir(self):
    logging.vlog(1, 'Creating common runfiles %s', self._common_runfiles)
    os.makedirs(self._common_runfiles, exist_ok=True)

  def package_target(self, target: str):
    """Package bazel target into output_dir.

    Args:
      target: Bazel target to package (ex. xls/tools/eval_ir_main)
    """
    # Search and replace leading // and : in t
    # this converts targets in bazel target specification
    # //a/b:c to the path where the target
    # can be found: a/b/c
    target = target.replace('//', '')
    target = target.replace(':', '/')

    # Determine files to copy via manifest
    target_exe = os.path.join(self._bazel_bin, target)
    target_manifest = '%s.runfiles_manifest' % target_exe
    target_runfiles = '%s.runfiles' % target_exe

    manifest = Manifest(global_manifest=self._global_manifest)
    manifest.read(target_manifest)

    # Now copy files
    self._copy_to_output(target_exe)
    for mf in manifest.all_files:
      mf_path = os.path.join(target_runfiles, mf)
      mf_mapping = manifest.get_mapping(mf)

      self._copy_to_output(mf_path, mf_mapping)

  def _copy_to_output(self, path: str, mapping: Optional[str] = None):
    """Copy a single file from a manifest to the output directory.

    When copying a file from a bazel build to the output directory, this
    method will either
      1) create a symlink to a file under .common_runfiles
      2) copy the file to the destination path.
    This mirrors the behavior of bazel which will
      1) create a symplink to a file under .bazel_cache
      2) create a file

    Args:
      path: File to copy
      mapping: Bazel cached file mapped to path
    """

    rel_path = os.path.relpath(path, start=self._bazel_bin)
    output_path = os.path.join(self._output_dir, rel_path)

    logging.debug('DEBUG: Copying %s mapping %s', rel_path, mapping)
    logging.debug('DEBUG:  - location %s', output_path)

    # Make directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if mapping is None:
      # Copy without linking as manifest does not have any mapping
      shutil.copy2(path, output_path, follow_symlinks=True)
    else:
      # Manifest has a mapping
      #   copy the file to .common_runfiles (if not already done)
      #   and create a symlink .
      link_to = self._copy_and_get_link(mapping)

      if link_to is None:
        # TODO(tedhong): 2020-10-01 Reduce occurrences of this
        logging.info(
            'INFO:  - Unable to find common path for %s, copying instead',
            mapping,
        )
        shutil.copy2(path, output_path, follow_symlinks=True)
      else:
        link_rel = os.path.relpath(link_to, os.path.dirname(output_path))
        logging.debug(
            'DEBUG:  - Linking %s to %s (%s)', output_path, link_rel, link_to
        )
        os.symlink(link_rel, output_path)
        pass

  def _copy_and_get_link(self, path: str):
    """Copies (if not previously done) a file to .common_runfiles.

    Args:
      path: File to copy

    Returns:
      detination path, None if unable to copy
    """
    # Determine which directory path is found under.
    # If unable to determine, return None so that
    # the program copies instead of links

    if path in self._linked_files:
      return self._linked_files[path]
    else:
      # See if path is under execroot or outputroot
      use_path = None

      if (
          os.path.commonpath([path, self._bazel_execroot])
          == self._bazel_execroot
      ):
        use_path = os.path.relpath(path, start=self._bazel_execroot)
        logging.debug('DEBUG:  - Under execroot - path %s', path)
      elif (
          os.path.commonpath([path, self._bazel_outputroot])
          == self._bazel_outputroot
      ):
        use_path = os.path.relpath(path, start=self._bazel_outputroot)
        logging.debug('DEBUG:  - Under outputroot - path %s', path)

      if use_path is not None:
        # Copy file to path
        dest_path = os.path.join(self._common_runfiles, use_path)

        if os.path.exists(dest_path):
          logging.fatal(
              'FATAL: Unable to copy %s to %s, file exists', path, dest_path
          )

        logging.debug('DEBUG: Link copying %s to %s', path, dest_path)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(path, dest_path, follow_symlinks=True)
        self._linked_files[path] = dest_path

        return dest_path
    return None


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Running XLS OSS packager')
  logging.info('  Output directory: %s', FLAGS.output_dir)
  logging.info('  Bazel bin dir: %s', FLAGS.bazel_bin)
  logging.info('  Bazel execroot: %s', FLAGS.bazel_execroot)
  logging.info('  Targets:')
  for t in FLAGS.inc_target:
    logging.info('   - %s', t)

  # Ensure that output directory is empty
  if os.path.exists(FLAGS.output_dir):
    if os.path.isfile(FLAGS.output_dir):
      logging.error('ERROR: --output_dir %s should be a directory')
      return -1

    if os.listdir(FLAGS.output_dir):
      logging.error('ERROR: --output_dir %s should be empty')
      return -1

  # Create output_dir and package each target
  packager = BazelTargetPackager(
      output_dir=FLAGS.output_dir,
      bazel_bin=FLAGS.bazel_bin,
      bazel_execroot=FLAGS.bazel_execroot,
  )

  packager.create_output_dir()
  for t in FLAGS.inc_target:
    logging.info('Packaging target %s...', t)
    packager.package_target(t)

  return 0


if __name__ == '__main__':
  app.run(main)
