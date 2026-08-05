import type { Edge } from "@xyflow/react";
import type { ArtifactInfo } from "../stores/run";

/**
 * Resolve the artifact flowing into a viewer node's body (issues.md L3):
 * the input edge (matched on the node's declared input port) names a
 * source node + source handle; only the artifact whose `handle` names
 * that same OUTPUT port feeds it. §7.2 makes the handle the authoritative
 * router — the old "source node's last artifact" fallback displayed the
 * wrong figure whenever a multi-output source (Align Runs, Estimate TF)
 * fed a viewer. Absent/"" handles pair only with absent/"" handles (the
 * L3-backend normalized contract), and the freshest matching artifact
 * wins (re-runs append).
 */
export function resolveInputArtifact(
  edges: Edge[],
  artifacts: ArtifactInfo[],
  nodeId: string,
  inputPort?: string,
): ArtifactInfo | null {
  const edge = edges.find(
    (e) =>
      e.target === nodeId &&
      (inputPort == null || (e.targetHandle ?? "") === inputPort),
  );
  if (!edge) return null;
  const want = edge.sourceHandle ?? "";
  const matching = artifacts.filter(
    (a) => a.nodeId === edge.source && (a.handle ?? "") === want,
  );
  return matching.length ? matching[matching.length - 1] : null;
}
