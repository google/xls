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

goog.module('xls.irVisualization');

const graphView = goog.require('xls.graphView');
const irGraph = goog.require('xls.irGraph');
const selectableGraph = goog.require('xls.selectableGraph');

/**
 * Returns the offset of the selection (cursor) within a text element.
 * TODO(meheff): Move this into a separate file and share with hls/xls/ui tool.
 * @param {!Element} node
 * @return {?number}
 */
function getOffsetWithin(node) {
  let sumPrevSiblings = (node) => {
    let total = 0;
    for (let sib of node.parentNode.childNodes) {
      if (sib === node) {
        break;
      }
      total += sib.textContent.length;
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
    if (!currentNode || currentNode === document) {
      return null;
    }
  }
  return offset;
}

/**
 * Sets the offset of the selection (cursor) within a text element.
 * TODO(meheff): Move this into a separate file and share with hls/xls/ui tool.
 * @param {!Element} node The text element.
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
    if (offset < textNode.textContent.length) {
      window.getSelection().setPosition(textNode, offset);
      return;
    } else {
      offset -= textNode.textContent.length;
    }
  }
}

/**
 * Class for visualizing IR graphs. Manages the text area containing the IR and
 * the element in which the graph is drawn.
 */
class IrVisualizer {
  /**
   * @param {!Element} graphElement DOM element to hold the graph.
   * @param {!Element} irElement Input DOM element holding IR text.
   * @param {?Element=} nodeMetadataElement DOM element to write node metadata
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
   */
  setSourceOkHandler(callback) {
    this.sourceOkCallback_ = callback;
  }

  /**
   * Sets the callback to call when the IR parsing encountered an error.
   * Callback takes a single argument, the error message.
   * @param {function(string)} callback
   */
  setSourceErrorHandler(callback) {
    this.sourceErrorCallback_ = callback;
  }

  /**
   * Selects the nodes on the critical path through the graph. All other nodes
   * are unselected.
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
    document.querySelector('.ir-identifier-' + nodeId)
        .classList.add('ir-identifier-highlighted');
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
      setInnerHtml(this.nodeMetadataElement_, text);
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
    document.querySelector('.ir-identifier-' + nodeId)
        .classList.remove('ir-identifier-highlighted');
    if (this.irGraph_ && this.nodeMetadataElement_) {
      this.nodeMetadataElement_.textContent = '';
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
    document.querySelector(`.ir-def-${srcId}`)
        .classList.add('ir-identifier-highlighted');
    document.querySelector(`.ir-use-${srcId}-${tgtId}`)
        .classList.add('ir-identifier-highlighted');
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
    document.querySelector(`.ir-def-${srcId}`)
        .classList.remove('ir-identifier-highlighted');
    document.querySelector(`.ir-use-${srcId}-${tgtId}`)
        .classList.remove('ir-identifier-highlighted');
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
        document.querySelector('.ir-identifier-' + change.id)
            .classList.remove('ir-identifier-selected');
      }
      if (change.to == selectableGraph.SelectState.SELECTED) {
        document.querySelector('.ir-identifier-' + change.id)
            .classList.add('ir-identifier-selected');
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
    document.querySelectorAll('.ir-identifier').forEach(elem => {
      elem.addEventListener('mouseenter', e => {
        self.highlightNode(
            /** @type {string} */ (e.target.dataset.irIdentifier));
      });
      elem.addEventListener('mouseleave', e => {
        self.unhighlightNode(
            /** @type {string} */ (e.target.dataset.irIdentifier));
      });
      elem.addEventListener('click', e => {
        let identifier = /** @type {string} */ (e.target.dataset.irIdentifier);
        if (e.ctrlKey && self.graphView_) {
          // If the control key is down. Zoom in on the node in the graph.
          self.graphView_.focusOnNode(identifier);
        } else {
          // Toggle the selection state.
          self.selectNode(identifier, !self.graph_.isNodeSelected(identifier));
        }
      });
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
    let text = this.irElement_.textContent;
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
    // matchAll is not yet recognized by the JS Compiler as it is ES2020.
    /** @suppress {missingProperties} */
    let matches = text.matchAll(/[a-zA-Z_][a-zA-Z0-9_.]*/g);
    for (const match of matches) {
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
    let focusOffset = getOffsetWithin(this.irElement_);
    setInnerHtml(this.irElement_, pieces.join(''));
    if (focusOffset != null) {
      setPositionAtOffset(this.irElement_, focusOffset);
    }
  }

  /**
   * Sends the IR to the server for parsing and, if successful, highlights the
   * IR.
   * @param {?function()} cb
   */
  parseAndHighlightIr(cb) {
    if (this.parseInFlight_) {
      return;
    }
    this.parseInFlight_ = true;
    let text = this.irElement_.textContent;
    let xmr = new XMLHttpRequest();
    xmr.open('POST', '/graph');
    let self = this;
    xmr.addEventListener('load', function() {
      if (xmr.status < 200 || xmr.status >= 400) {
        return;
      }

      // TODO: define a type for the graph object.
      let response = /** @type {!Object} */ (JSON.parse(xmr.responseText));
      self.parseInFlight_ = false;
      if (self.irElement_.textContent != text) {
        return self.parseAndHighlightIr(cb);
      }
      // Clear graph.
      self.irGraph_ = null;
      self.graph_ = null;
      if (self.graphView_) {
        self.graphView_.destroy();
        self.graphView_ = null;
      }

      if (response['error_code'] == 'ok') {
        if (self.sourceOkCallback_) {
          self.sourceOkCallback_();
        }
        // The returned JSON in the 'graph' key is a JSON object whose structure
        // is defined by the xls.visualization.Package proto. The particular
        // function to view is named in the 'entry' field of the package proto.
        let graph = null;
        for (let func of response['graph']['function_bases']) {
          if (response['graph']['entry'] == func['name']) {
            graph = func;
          }
        }
        if (graph == null) {
          return;
        }
        self.irGraph_ = new irGraph.IrGraph(graph);
        self.graph_ = new selectableGraph.SelectableGraph(self.irGraph_);

        self.highlightIr_(graph);
        self.setIrTextListeners_();
      } else {
        if (!!self.sourceErrorCallback_) {
          self.sourceErrorCallback_(response['message']);
        }
        let focusOffset = getOffsetWithin(self.irElement_);
        setInnerHtml(self.irElement_, self.irElement_.textContent);
        if (focusOffset) {
          setPositionAtOffset(self.irElement_, focusOffset);
        }
      }
      if (cb) {
        cb();
      }
    });
    let data = new FormData();
    data.append('text', text);
    xmr.send(data);
  }

  /**
   * Creates and renders a graph of the IR using Cytoscape. This should only be
   * called after the IR has been successfully parsed and the SelectionGraph
   * constructed.
   * @param {boolean} showOnlySelected Whether to include only selected and
   *     frontier nodes and edges in the graph.
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
          document.querySelector(`.ir-def-${nodeId}`).scrollIntoView();
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

exports = {IrVisualizer};
