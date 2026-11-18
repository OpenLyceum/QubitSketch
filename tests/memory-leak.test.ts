/**
 * memory-leak.test.ts
 *
 * Creates the model and the GatePalettePanel, disposes them, forces garbage
 * collection, then asserts via WeakRef that the disposed objects were actually
 * collected. Requires --expose-gc (configured in vitest.config.ts execArgv).
 *
 * V8 requires a *function* boundary (not merely a block scope) to release local
 * variables for collection, so every allocation is wrapped in a helper function
 * whose strong references die when it returns.
 *
 * GatePalettePanel is the only sim component that adds/removes nodes at runtime
 * (drag previews in the dragLayer, hover tooltips in the overlayLayer) and links
 * to the shared model's selectedToolProperty. dispose() must unlink that
 * listener, dispose the drag listeners, and dispose the whole child subtree so
 * the global QubitSketchColors Properties no longer retain the panel.
 */

import { Vector2 } from "scenerystack/dot";
import { Node } from "scenerystack/scenery";
import { describe, expect, it } from "vitest";
import { GATE_MATRICES } from "../src/circuit-screen/model/GateMatrices.js";
import { QubitSketchModel } from "../src/circuit-screen/model/QubitSketchModel.js";
import type { SlotDropTarget } from "../src/circuit-screen/view/CircuitCanvas.js";
import { GatePalettePanel, type PaletteDragContext } from "../src/circuit-screen/view/GatePalettePanel.js";
import { MatrixTooltipNode } from "../src/circuit-screen/view/MatrixTooltipNode.js";
import { StringManager } from "../src/i18n/StringManager.js";
import { describeDisposalLeaks, expectNoListenerGrowth, forceGC } from "./helpers/memoryLeak.js";

// Long-lived model sentinel: stays alive across tests so that a disposed panel
// must drop its own reference (unlink selectedToolProperty) to be collectible.
const sharedModel = new QubitSketchModel();

// Drop target that never reports a slot — drags simply cancel.
const noopDropTarget: SlotDropTarget = {
  slotIndexAt: (_globalPoint: Vector2, _qubitCount: number) => null,
};

/** A fresh drag context (drag + overlay layers) for one panel under test. */
function makeDragContext(): PaletteDragContext {
  return { dragLayer: new Node(), dropTarget: noopDropTarget, overlayLayer: new Node() };
}

/** Create a panel (with or without drag support), dispose it, return WeakRefs. */
function createAndDisposePanel(withDragContext: boolean): {
  panelRef: WeakRef<object>;
  dragContext: PaletteDragContext | undefined;
} {
  const dragContext = withDragContext ? makeDragContext() : undefined;
  const panel = new GatePalettePanel(sharedModel, dragContext);
  const panelRef = new WeakRef<object>(panel);
  panel.dispose();
  return { panelRef, dragContext };
}

describe("Memory leak regression", () => {
  it("bare QubitSketchModel is collected (no global retention)", async () => {
    const ref = (() => new WeakRef<object>(new QubitSketchModel()))();
    await forceGC(ref);
    expect(ref.deref()).toBeUndefined();
  });

  it("GatePalettePanel (drag-enabled) is collected after dispose", async () => {
    const { panelRef, dragContext } = createAndDisposePanel(true);
    await forceGC(panelRef);
    expect(panelRef.deref()).toBeUndefined();
    // dispose must not leave preview/tooltip nodes stranded in the layers.
    expect(dragContext?.dragLayer.children.length).toBe(0);
    expect(dragContext?.overlayLayer.children.length).toBe(0);
  });

  it("GatePalettePanel (click-to-select, no dragContext) is collected after dispose", async () => {
    const { panelRef } = createAndDisposePanel(false);
    await forceGC(panelRef);
    expect(panelRef.deref()).toBeUndefined();
  });

  it("double dispose() does not throw", () => {
    const panel = new GatePalettePanel(sharedModel, makeDragContext());
    panel.dispose();
    expect(() => panel.dispose()).not.toThrow();
  });

  it("repeated create/dispose cycles leave no survivors", async () => {
    const refs: WeakRef<object>[] = [];
    for (let i = 0; i < 20; i++) {
      refs.push(createAndDisposePanel(i % 2 === 0).panelRef);
    }
    await forceGC(refs);
    const survivors = refs.filter((r) => r.deref() !== undefined).length;
    expect(survivors).toBe(0);
  });
});

describeDisposalLeaks([
  {
    name: "GatePalettePanel (drag-enabled)",
    create: () => new GatePalettePanel(sharedModel, makeDragContext()),
    idempotentDispose: true,
  },
  {
    name: "GatePalettePanel (click-to-select)",
    create: () => new GatePalettePanel(sharedModel),
    idempotentDispose: true,
  },
]);

// GatePalettePanel builds a new MatrixTooltipNode on every palette hover and disposes it on exit
// (or on the next hover). The tooltip's description Text links to a localized string Property
// that outlives the panel, so a tooltip whose content is not disposed stays reachable from it.
const descriptions = StringManager.getInstance().getToolDescriptions();

describeDisposalLeaks([
  {
    name: "MatrixTooltipNode (gate matrix)",
    create: () => new MatrixTooltipNode(GATE_MATRICES.H, descriptions.H),
    keepAlive: () => [descriptions.H],
  },
  {
    name: "MatrixTooltipNode (marker, no matrix)",
    create: () => new MatrixTooltipNode(null, descriptions.eraser),
    keepAlive: () => [descriptions.eraser],
  },
]);

describe("palette hover tooltips", () => {
  type HoverListener = { enter?: (event: unknown) => void; exit?: (event: unknown) => void };

  /** The palette hit-areas (one PDOM button per tool), in ALL_TOOLS order. */
  function hitAreas(panel: Node): Node[] {
    const areas: Node[] = [];
    const visit = (node: Node): void => {
      if (node.tagName === "button") {
        areas.push(node);
      }
      node.children.forEach(visit);
    };
    visit(panel);
    return areas;
  }

  function hover(area: Node, phase: "enter" | "exit"): void {
    const event = { pointer: { point: new Vector2(10, 10) } };
    // Only the palette's own plain-object hover listener; DragListener has its own enter/exit.
    for (const listener of area.inputListeners as HoverListener[]) {
      if (Object.getPrototypeOf(listener) === Object.prototype) {
        listener[phase]?.(event);
      }
    }
  }

  it("entering and leaving tool buttons keeps description listener counts flat", async () => {
    const dragContext = makeDragContext();
    const panel = new GatePalettePanel(sharedModel, dragContext);
    const areas = hitAreas(panel);
    const [hArea, xArea, eraserArea] = [areas[0]!, areas[1]!, areas.at(-1)!];
    await expectNoListenerGrowth([descriptions.H, descriptions.X, descriptions.eraser], () => {
      hover(hArea, "enter");
      hover(xArea, "enter"); // replaces the H tooltip without an exit
      hover(xArea, "exit");
      hover(eraserArea, "enter");
      hover(eraserArea, "exit");
    });
    expect(dragContext.overlayLayer.children).toHaveLength(0);
    panel.dispose();
  });

  it("the previous tooltip and its content are collected after the pointer leaves", async () => {
    const dragContext = makeDragContext();
    const panel = new GatePalettePanel(sharedModel, dragContext);
    const area = hitAreas(panel)[0]!;
    const hoverOnce = (): WeakRef<object>[] => {
      hover(area, "enter");
      const tooltip = dragContext.overlayLayer.children[0]!;
      expect(tooltip).toBeInstanceOf(MatrixTooltipNode);
      const refs = [new WeakRef<object>(tooltip)];
      const visit = (node: Node): void => {
        refs.push(new WeakRef<object>(node));
        node.children.forEach(visit);
      };
      tooltip.children.forEach(visit);
      hover(area, "exit");
      return refs;
    };
    const refs = [...hoverOnce(), ...hoverOnce()];
    await forceGC(refs);
    expect(refs.filter((ref) => ref.deref() !== undefined).length, "tooltip nodes retained").toBe(0);
    panel.dispose();
  });
});
