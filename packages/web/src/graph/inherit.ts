import type { Edge, Node } from "@xyflow/react";
import type { MTNodeData } from "../stores/workflow";
import type { InheritMap, NodeManifest } from "../api/client";

/**
 * Chain-walking inheritance resolver (DESIGN.md §4 "Chain-walking inheritance").
 *
 * `inherit` is a pure, frontend-only graph walk over the workflow document. For
 * a target node + param it returns the candidate value(s) supplied by the
 * connected upstream chain, so identities (survey/station/run, ...) flow
 * through arbitrary intermediate processing nodes without losing information.
 *
 * Rules:
 *  - each node declares only its own `inherit: {input_port: {target: upstream}}`
 *  - a node's own non-empty param value wins; otherwise:
 *      * if the node declares an inherit mapping for the param, follow exactly
 *        those input port(s) and seek the *mapped* upstream param name
 *      * else (pass-through node, e.g. a filter with no inherit declaration)
 *        walk every input port seeking the *same* name — this is what makes
 *        chain-walking transitive across intermediate nodes
 *  - a visited set guards cycles; candidates from parallel sources are unioned
 */

export interface InheritContext {
  nodes: Node<MTNodeData>[];
  edges: Edge[];
  manifests: Map<string, NodeManifest>; // node type -> manifest
}

export function paramOf(node: Node<MTNodeData> | undefined, name: string): unknown {
  return (node?.data as MTNodeData | undefined)?.params?.[name];
}

export function nonEmpty(value: unknown): boolean {
  return value !== undefined && value !== null && String(value).trim() !== "";
}

/** The set of param names a node declares as inheritable (any input port).
 *  Only these may be auto-filled / rendered as dropdowns; all other params
 *  stay fully manual (root-node defaults, radio/select config, ...). */
export function inheritableParams(manifest: NodeManifest | undefined): Set<string> {
  const set = new Set<string>();
  for (const mapping of Object.values(manifest?.inherit ?? {})) {
    for (const target of Object.keys(mapping)) set.add(target);
  }
  return set;
}

/** Candidate *inherited* values for `paramName` on `nodeId`, walking the
 *  upstream chain. Does NOT include `nodeId`'s own value, so a param only
 *  renders as a dropdown / auto-fills when an upstream node actually provides
 *  a value — otherwise it stays a normal, editable field. The recursion
 *  (`includeOwn=true`) does consider each upstream node's own value.
 */
export function resolveInherited(
  ctx: InheritContext,
  nodeId: string,
  paramName: string,
  visited: Set<string> = new Set(),
  includeOwn = false,
): string[] {
  const key = `${nodeId}::${paramName}`;
  if (visited.has(key)) return [];
  visited.add(key);

  const node = ctx.nodes.find((n) => n.id === nodeId);
  if (!node) return [];
  const manifest = ctx.manifests.get((node.data as MTNodeData).nodeType);
  const inherit = manifest?.inherit as InheritMap | undefined;

  const out = new Set<string>();

  // intermediate hop: a value actually set on this node is a candidate
  if (includeOwn) {
    const own = paramOf(node, paramName);
    if (nonEmpty(own)) {
      out.add(String(own));
      return [...out];
    }
  }

  const portNames = (manifest?.inputs ?? []).map((i) => i.name);
  const covers = Object.values(inherit ?? {}).some((m) => m[paramName] !== undefined);

  const seek = (portName: string, seekName: string) => {
    for (const edge of ctx.edges) {
      if (edge.target !== nodeId || edge.targetHandle !== portName) continue;
      for (const v of resolveInherited(ctx, edge.source, seekName, visited, true)) {
        out.add(v);
      }
    }
  };

  if (covers && inherit) {
    // respect the declared mapping: walk only the mapped ports
    for (const [portName, mapping] of Object.entries(inherit)) {
      const seekName = mapping[paramName];
      if (seekName) seek(portName, seekName);
    }
  } else {
    // pass-through fallback: walk every input port with the same name
    for (const portName of portNames) seek(portName, paramName);
  }

  return [...out];
}

/** Build an InheritContext from a catalog manifest list. */
export function makeInheritContext(
  manifests: NodeManifest[] | undefined,
  nodes: Node<MTNodeData>[],
  edges: Edge[],
): InheritContext {
  const map = new Map<string, NodeManifest>();
  for (const m of manifests ?? []) map.set(m.type, m);
  return { nodes, edges, manifests: map };
}
