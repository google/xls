/**
 * Copyright 2020 The XLS Authors
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

goog.module('xls.IrVisualization');

var graphView = goog.require('xls.graphView');
var irGraph = goog.require('xls.irGraph');
var selectableGraph = goog.require('xls.selectableGraph');

/**
 * Returns the offset of the selection (cursor) within a text element.
 * TODO(meheff): Move this into a separate file and share with hls/xls/ui tool.
 * @param {!Object} node
 * @return {?number}
 */
function getOffsetWithin(node) {
  let sumPrevSiblings = (node) => {
    let prevSiblings = $(node.parentNode).contents();
    let total = 0;
    for (let sib of /** @type {!Array<!Object>} */ (prevSiblings.toArray())) {
      if (sib === node) {
        break;
      }
      total += $(sib).text().length;
    }
    return total;
  };

  let sel = window.getSelection();
  let offset = sel.focusOffset;
  let currentNode = sel.focusNode;
  if (!currentNode) {
    return null;
  }
  while (currentNode !== node) {
    let prevSiblings = sumPrevSiblings(currentNode);
    offset += prevSiblings;
    currentNode = currentNode.parentNode;
    if (!currentNode) {
      return null;
    }
  }
  return offset;
}

/**
 * Sets the offset of the selection (cursor) within a text element.
 * TODO(meheff): Move this into a separate file and share with hls/xls/ui tool.
 * @param {!Object} node The text element.
 * @param {number} offset The offset within the text element.
 */
function setPositionAtOffset(node, offset) {
  let allTextNodesUnder = (node) => {
    let result = [];
    for (let child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        result.push(child);
      } else {
        result = result.concat(allTextNodesUnder(child));
      }
    }
    return result;
  };

  let textNodes = allTextNodesUnder(node);
  for (let textNode of textNodes) {
    if (offset < textNode.length) {
      window.getSelection().setPosition(textNode, offset);
      return;
    } else {
      offset -= textNode.length;
    }
  }
}

/**
 * Class for visualizing IR graphs. Manages the text area containing the IR and
 * the element in which the graph is drawn.
 */
class IrVisualizer {
  /**
   * @param {!Object} graphElement DOM element to hold the graph.
   * @param {!Object} irElement Input DOM element holding IR text.
   * @param {?Object=} nodeMetadataElement DOM element to write node metadata
   *     text into.
   */
  constructor(graphElement, irElement, nodeMetadataElement = undefined) {
    this.graphElement_ = graphElement;
    this.irElement_ = irElement;
    this.nodeMetadataElement_ = nodeMetadataElement;

    /**
     * The graph view object.
     * @private {?graphView.GraphView}
     */
    this.graphView_ = null;

    /**
     * @private {?irGraph.IrGraph}
     */
    this.irGraph_ = null;

    /**
     * The SelectableGraph object.
     * @private {?selectableGraph.SelectableGraph}
     */
    this.graph_ = null;

    /**
     * Whether a parse request is in flight. Only a single request is allowed at
     * a time.
     * @private {boolean}
     */
    this.parseInFlight_ = false;

    /**
     * Callbacks for success/error status when parsing and graphifying the IR.
     * @private {function()|undefined}
     */
    this.sourceOkCallback_ = undefined;

    /** @private {function(string)|undefined} */
    this.sourceErrorCallback_ = undefined;
  }

  /**
   * Sets the callback to call when the IR is successfully parsed.
   * @param {function()} callback
   * @export
   */
  setSourceOkHandler(callback) {
    this.sourceOkCallback_ = callback;
  }

  /**
   * Sets the callback to call when the IR parsing encountered an error.
   * Callback takes a single argument, the error message.
   * @param {function(string)} callback
   * @export
   */
  setSourceErrorHandler(callback) {
    this.sourceErrorCallback_ = callback;
  }

  /**
   * Selects the nodes on the critical path through the graph. All other nodes
   * are unselected.
   * @export
   */
  selectCriticalPath() {
    let criticalPathNodeIds =
        this.irGraph_.nodes()
            .filter((n) => 'on_critical_path' in n.attributes)
            .map((n) => n.id);
    this.applyChange_(this.graph_.selectOnlyNodes(criticalPathNodeIds));
  }

  /**
   * Sets whether to show only the selected and frontier node and elements in
   * the graph depending on parameter 'value'.
   * @param {boolean} value
   * @export
   */
  setShowOnlySelected(value) {
    if (this.graphView_) {
      this.graphView_.setShowOnlySelected(value);
    }
  }

  /**
   * Highlights the node with the given id in the graph and the textual IR.
   * @param {string} nodeId
   */
  highlightNode(nodeId) {
    $('.ir-identifier-' + nodeId).addClass('ir-identifier-highlighted');
    if (this.irGraph_ && this.nodeMetadataElement_) {
      let text = '<b>node:</b> ' + this.irGraph_.node(nodeId).ir;
      let delay = this.irGraph_.node(nodeId).attributes['delay_ps'];
      if (delay != null) {
        text += '\n<b>delay:</b> ' + delay + 'ps';
      }
      let known = this.irGraph_.node(nodeId).attributes['known_bits'];
      if (known) {
        text += '\n<b>known bits:</b> ' + known;
      }
      let cycle = this.irGraph_.node(nodeId).attributes['cycle'];
      if (cycle != undefined) {
        text += '\n<b>cycle:</b> ' + cycle;
      }
      this.nodeMetadataElement_.html(text);
    }
    if (this.graphView_) {
      this.graphView_.highlightNode(nodeId);
    }
  }

  /**
   * Unhighlights the node with the given id in the graph and the textual IR.
   * @param {string} nodeId
   */
  unhighlightNode(nodeId) {
    $('.ir-identifier-' + nodeId).removeClass('ir-identifier-highlighted');
    if (this.irGraph_ && this.nodeMetadataElement_) {
      this.nodeMetadataElement_.text('');
    }
    if (this.graphView_) {
      this.graphView_.unhighlightNode(nodeId);
    }
  }

  /**
   * Highlights the edge with the given id in the graph and the textual IR.
   * @param {string} edgeId
   */
  highlightEdge(edgeId) {
    let srcId = this.irGraph_.edge(edgeId).source;
    let tgtId = this.irGraph_.edge(edgeId).target;
    $(`.ir-def-${srcId}`).addClass('ir-identifier-highlighted');
    $(`.ir-use-${srcId}-${tgtId}`).addClass('ir-identifier-highlighted');
    if (this.graphView_) {
      this.graphView_.highlightEdge(edgeId);
    }
  }

  /**
   * Highlights the edge with the given id in the graph and the textual IR.
   * @param {string} edgeId
   */
  unhighlightEdge(edgeId) {
    let srcId = this.irGraph_.edge(edgeId).source;
    let tgtId = this.irGraph_.edge(edgeId).target;
    $(`.ir-def-${srcId}`).removeClass('ir-identifier-highlighted');
    $(`.ir-use-${srcId}-${tgtId}`).removeClass('ir-identifier-highlighted');
    if (this.graphView_) {
      this.graphView_.unhighlightEdge(edgeId);
    }
  }

  /**
   * Applies the change object as returned by SelectionGraph::computeChanges_
   * to the textual IR and graph. Selects and deselects nodes and edges as well
   * as adds/removes elements if showOnlySelected is set.
   * @param {!selectableGraph.SelectionChangeSet} changes
   * @private
   */
  applyChange_(changes) {
    // (De)select identifier elements in the textual IR.
    for (const change of changes.nodes) {
      if (change.from == selectableGraph.SelectState.SELECTED) {
        $('.ir-identifier-' + change.id).removeClass('ir-identifier-selected');
      }
      if (change.to == selectableGraph.SelectState.SELECTED) {
        $('.ir-identifier-' + change.id).addClass('ir-identifier-selected');
      }
    }

    if (this.graphView_) {
      this.graphView_.applyChange(changes);
    }
  }

  /**
   * Set the selection state of the node with the given id to 'value'.
   * @param {string} nodeId
   * @param {boolean} value
   */
  selectNode(nodeId, value) {
    this.applyChange_(this.graph_.selectNode(nodeId, value));
  }

  /**
   * Sets various listeners for hovering over and selecting identifiers in the
   * IR text.
   */
  setIrTextListeners_() {
    let self = this;
    $('.ir-identifier').mouseenter(function() {
      self.highlightNode(/** @type {string} */ ($(this).data('ir-identifier')));
    });
    $('.ir-identifier').mouseleave(function() {
      self.unhighlightNode(
          /** @type {string} */ ($(this).data('ir-identifier')));
    });
    $('.ir-identifier').click(function(e) {
      let identifier = /** @type {string} */ ($(this).data('ir-identifier'));
      if (e.originalEvent.ctrlKey && self.graphView_) {
        // If the control key is down. Zoom in on the node in the graph.
        self.graphView_.focusOnNode(identifier);
      } else {
        // Toggle the selection state.
        self.selectNode(identifier, !self.graph_.isNodeSelected(identifier));
      }
    });
  }

  /**
   * Highlights the IR text source using the JSON graphified IR. Currently just
   * makes the identifiers bold and wraps them in spans.
   * TODO(meheff): This should probably be done on the server side.
   * @param {!Object} jsonGraph
   * @private
   */
  highlightIr_(jsonGraph) {
    let text = this.irElement_.text();
    // A map containing the node identifiers in the graph. The value of the map
    // is a count as the identifiers are encountered when walking through the
    // IR. It is used to identify defs (first encounter of an identifier).
    let identifiers = {};
    let nameToId = {};
    jsonGraph['nodes'].forEach(function(node, index) {
      identifiers[node.name] = 0;
      nameToId[node.name] = node.id;
    });
    let pieces = [];
    let lastPieceEnd = 0;
    let lastDef = undefined;
    for (const match of text.matchAll(/[a-zA-Z_][a-zA-Z0-9_.]*/g)) {
      if (match.index > lastPieceEnd) {
        pieces.push(text.slice(lastPieceEnd, match.index));
      }
      if (match[0] in identifiers) {
        let id = nameToId[match[0]];
        let classes = ['ir-identifier', 'ir-identifier-' + id];
        // If this is the first time we've seen the identifier it is a def.
        let isDef = identifiers[match[0]] == 0;
        identifiers[match[0]] += 1;
        // To enable highlighting of edge end points, add classes for the defs
        // and uses of values.
        if (isDef) {
          classes.push('ir-def-' + id);
          lastDef = id;
        } else if (lastDef) {
          classes.push(`ir-use-${id}-${lastDef}`);
        }
        pieces.push(`<span class="${classes.join(' ')}" data-ir-identifier="${
            id}">${match[0]}</span>`);
      } else {
        pieces.push(match[0]);
      }
      lastPieceEnd = match.index + match[0].length;
    }
    pieces.push(text.slice(lastPieceEnd));
    let focusOffset = getOffsetWithin(this.irElement_.get(0));
    this.irElement_.html(pieces.join(''));
    if (focusOffset != null) {
      setPositionAtOffset(this.irElement_.get(0), focusOffset);
    }
  }

  /**
   * Sends the IR to the server for parsing and, if successful, highlights the
   * IR.
   * @export
   */
  parseAndHighlightIr(cb) {
    if (this.parseInFlight_) {
      return;
    }
    this.parseInFlight_ = true;
    let text = this.irElement_.text();
    $.post('/graph', {text: text}, (response_str) => {
      // TODO: define a type for the graph object.
      let response = /** @type {!Object} */ (JSON.parse(response_str));
      this.parseInFlight_ = false;
      if (this.irElement_.text() != text) {
        return this.parseAndHighlightIr(cb);
      }
      // Clear graph.
      this.irGraph_ = null;
      this.graph_ = null;
      if (this.graphView_) {
        this.graphView_.destroy();
        this.graphView_ = null;
      }

      if (response['error_code'] == 'ok') {
        if (this.sourceOkCallback_) {
          this.sourceOkCallback_();
        }
        this.irGraph_ = new irGraph.IrGraph(response['graph']);
        this.graph_ = new selectableGraph.SelectableGraph(this.irGraph_);

        this.highlightIr_(response['graph']);
        this.setIrTextListeners_();
      } else {
        if (!!this.sourceErrorCallback_) {
          this.sourceErrorCallback_(response['message']);
        }
        let focusOffset = getOffsetWithin(this.irElement_.get(0));
        this.irElement_.html(this.irElement_.text());
        if (focusOffset) {
          setPositionAtOffset(this.irElement_.get(0), focusOffset);
        }
      }
      if (cb) {
        cb();
      }
    }, 'text');
  }

  /**
   * Creates and renders a graph of the IR using Cytoscape. This should only be
   * called after the IR has been successfully parsed and the SelectionGraph
   * constructed.
   * @param {boolean} showOnlySelected Whether to include only selected and
   *     frontier nodes and edges in the graph.
   * @export
   */
  draw(showOnlySelected) {
    if (!this.graph_) {
      console.log(
          'SelectableGraph not yet constructed. Parse error or slow response?');
      return;
    }
    this.graphView_ = new graphView.GraphView(
        this.graph_, this.graphElement_, showOnlySelected);

    this.graphView_.setHoverOnNodeCallback((nodeId, hoverOn) => {
      if (hoverOn) {
        this.highlightNode(nodeId);
      } else {
        this.unhighlightNode(nodeId);
      }
    });
    this.graphView_.setHoverOnEdgeCallback((edgeId, hoverOn) => {
      if (hoverOn) {
        this.highlightEdge(edgeId);
      } else {
        this.unhighlightEdge(edgeId);
      }
    });
    this.graphView_.setClickCallback((nodeId, ctrlPressed) => {
      if (nodeId) {
        if (ctrlPressed) {
          // Scroll the node into view in the IR text window.
          $(`.ir-def-${nodeId}`)[0].scrollIntoView();
        } else {
          // Toggle the selection state.
          this.selectNode(nodeId, !this.graph_.isNodeSelected(nodeId));
        }
      } else {
        // Clicking on the graph outside of a node (nodeId == null) clears the
        // selection state.
        this.applyChange_(this.graph_.selectOnlyNodes([]));
      }
    });
  }
}

goog.exportSymbol('xls.IrVisualizer', IrVisualizer);
exports = {IrVisualizer};
