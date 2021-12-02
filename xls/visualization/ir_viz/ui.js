/**
 * Copyright 2021 The XLS Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

goog.module('xls.ui');

const irVisualization = goog.require('xls.irVisualization');

let selectedOptLevel;
let selectedExample;
let visualizer;

/**
 * Returns the "Load Example" button UI element.
 * @return {!Element}
 */
function getLoadExampleButton() {
  return /** @type {!Element} */ (document.getElementById('load-example-btn'));
}

/**
 * Enables the given element (e.g., a button).
 * @param {!Element} elem
 */
function enable(elem) {
  elem.classList.remove('disabled');
  elem.removeAttribute('disabled');
}

/**
 * Disables the given element (e.g., a button).
 * @param {!Element} elem
 */
function disable(elem) {
  elem.classList.add('disabled');
  elem.setAttribute('disabled', true);
}

/**
 * Sets the optimization level according to the given selected value from the UI
 * element.
 * @param {number} level
 */
function selectBenchmarkOptLevel(level) {
  if (level == '') {
    selectedOptLevel = undefined;
    disable(getLoadExampleButton());
    return;
  }
  if (selectedExample != undefined) {
    enable(getLoadExampleButton());
  }
  selectedOptLevel = level;
}

/**
 * Sets the benchmark to the given example.
 * @param {string} example
 */
function selectBenchmark(example) {
  if (example == '') {
    selectedExample = undefined;
    disable(getLoadExampleButton());
    return;
  }
  if (selectedOptLevel != undefined) {
    enable(getLoadExampleButton());
  }
  selectedExample = example;
}

/**
 * Sets the IR file to the given example.
 * @param {string} example
 */
function selectIrFile(example) {
  if (example != '') {
    loadExample(example, undefined);
  }
}

/**
 * Loads the IR example with the given name and opt_level from the server.
 * @param {string=} name
 * @param {number=} opt_level
 */
function loadExample(name, opt_level) {
  let url = `/examples/${name}`;
  if (opt_level != undefined) {
    url += `?opt_level=${opt_level}`;
  }
  let xmr = new XMLHttpRequest();
  xmr.open('GET', url);
  xmr.addEventListener('load', function() {
    if (xmr.status < 200 || xmr.status >= 400) {
      return;
    }
    document.getElementById('ir-source-text').textContent = xmr.responseText;
    inputChangeHandler(() => visualizer.draw(showOnlySelectedState()));
  });
  xmr.send();
}

/**
 * Loads the selected IR example.
 */
function loadExampleButtonHandler() {
  loadExample(selectedExample, selectedOptLevel);
}

/**
 * Change handler for the IR text element.
 * @param {function()=} cb
 */
function inputChangeHandler(cb) {
  document.getElementById('view-graph-btn').disabled = true;
  document.getElementById('view-critical-path-btn').disabled = true;
  visualizer.parseAndHighlightIr(cb);
}

/**
 * Returns  show-only-selected state of the UI element.
 * @return {boolean}
 */
function showOnlySelectedState() {
  return document.getElementById('only-selected-checkbox').checked;
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('ir-upload').addEventListener('input', e => {
    if (e.target.files.length > 0) {
      e.target.files[0].text().then(function(text) {
        document.getElementById('ir-source-text').textContent = text;
      });
      inputChangeHandler();
    }
  });

  visualizer = new irVisualization.IrVisualizer(
      /** @type {!Element} */ (document.getElementById('graph')),
      /** @type {!Element} */ (document.getElementById('ir-source-text')),
      /** @type {!Element} */ (document.getElementById('function-select')),
      /** @type {!Element} */ (document.getElementById('node-metadata')));
  visualizer.setSourceOkHandler(function() {
    let src_status = document.getElementById('source-status');
    src_status.classList.remove('alert-danger', 'alert-dark');
    src_status.classList.add('alert-success');
    src_status.textContent = 'OK';
    document.getElementById('view-graph-btn').disabled = false;
    document.getElementById('view-critical-path-btn').disabled = false;
  });
  visualizer.setSourceErrorHandler(function(errorText) {
    let src_status = document.getElementById('source-status');
    src_status.classList.remove('alert-dark', 'alert-success');
    src_status.classList.add('alert-danger');
    src_status.textContent = errorText;
    document.getElementById('view-graph-btn').disabled = true;
    document.getElementById('view-critical-path-btn').disabled = true;
  });

  document.getElementById('ir-source-text').addEventListener('input', e => {
    inputChangeHandler();
  });

  document.getElementById('view-graph-btn').addEventListener('click', e => {
    visualizer.draw(showOnlySelectedState());
  });
  document.getElementById('view-critical-path-btn')
      .addEventListener('click', e => {
        visualizer.selectCriticalPath();
      });

  document.getElementById('only-selected-checkbox')
      .addEventListener('input', e => {
        visualizer.setShowOnlySelected(showOnlySelectedState());
      });

  // If --preload_ir_path was passed in to the server, the ir-source-text
  // element will be filled in with IR text. Trigger the inputChangedHandler
  // to parse it, select the critical path and show the graph.
  if (document.getElementById('ir-source-text').textContent.trim() != '') {
    inputChangeHandler(() => {
      visualizer.selectCriticalPath();
      visualizer.draw(showOnlySelectedState());
    });
  }

  // Pressing 'j' or 'k' will move up and down through the list of IR files
  // in the example directory (only enabled if --example_ir_dir given to
  // app.py). The common use case for --example_ir_dir is exposing the IR
  // after each pass in the optimization pipeline (created with opt_main
  // --ir_dump_path) so the j/k short cuts enable quick movement forward and
  // backwards through the optimization pipeline.
  document.addEventListener('keydown', event => {
    let file_selector = document.getElementById('ir-file-select');
    if (file_selector == undefined ||
        event.target == document.getElementById('ir-source-text') ||
        event.key != 'j' && event.key != 'k') {
      return;
    }
    let idx = file_selector.selectedIndex;
    if (event.key == 'j') {
      idx++;
    } else {
      idx--;
    }
    idx = idx < 1 ? 1 : idx;
    idx = idx >= file_selector.length ? file_selector.length - 1 : idx;
    file_selector.selectedIndex = idx;
    // TODO(meheff): Selected nodes should be preserved across transitions.
    selectIrFile(file_selector.value);
  });
});

// These functions are called directly from the UI elements (e.g., onclick
// handlers).
goog.exportSymbol('selectBenchmarkOptLevel', selectBenchmarkOptLevel);
goog.exportSymbol('selectBenchmark', selectBenchmark);
goog.exportSymbol('selectIrFile', selectIrFile);
goog.exportSymbol('loadExampleButtonHandler', loadExampleButtonHandler);
