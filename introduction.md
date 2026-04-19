# Introduction: The Data Structure of emptyUI

emptyUI treats a processing campaign as a directed acyclic graph of
declarations, not as a script. Everything the system knows about a pipeline —
which steps exist, what they consume, what they produce, and how results
flow between them — is carried by one portable JSON document, the
*workflow*. It contains a `nodes` list, in which every entry pairs a
namespaced node *type* (e.g. `core.io.load_csv`,
`aurora.process.estimate_tf`) with its concrete parameter values, and an
`edges` list, in which every connection names the source node, the source
port, and the target port by *name* rather than by position. Naming ports
explicitly makes the document resilient to refactoring: nodes may be
reordered or their ports reshuffled without invalidating saved workflows,
and node positions are UI-only metadata that headless execution ignores.

The static half of the data model is the *node manifest*, exported by the
registry and consumed verbatim by the front end. A manifest declares typed
input and output ports, a parameter schema with widget vocabulary, defaults
and ranges, and the rendering conveniences (`cascade`, `enabled_when`,
`inherit`) that let the interface preset channel names, gate widgets, and
propagate station identity upstream without hardcoding any node behavior.
Because the manifest is the single source of truth, adding a node type
requires no frontend change.

At run time the graph comes alive through *typed handles*. Nodes never pass
payloads to each other directly; the executor registers each declared
output in a per-run data store keyed by an opaque handle ID, and downstream
nodes receive the dereferenced values as keyword arguments. The handle
layer keeps WebSocket messages small, lets identical structures share cache
entries across runs, and lets the executor police a strict immutability
contract — nodes must be functional, never mutating their inputs — because
a payload may simultaneously back several branches and survive into later
runs through the cache.

The payload types themselves mirror the mth5/aurora ecosystem one-to-one.
An open archive (`mth5`) carries a run summary whose per-run extents drive
the interface's time selector. Time-series currency flows as `ts.run` (an
mth5 `RunTS` — a channel-labelled xarray dataset carrying units, filters
and provenance in its attrs), as single `ts.channel` components, or as a
`ts.runs` collection pairing member payloads with extent rows for
burst-mode archives. The frequency domain speaks `fc`: per decimation
level, a Dataset whose windows carry epoch-anchored time coordinates,
per-window `run_id` identity, and attributes recording the window schedule,
applied calibration, unit scaling and merge-gap provenance. Estimation
yields per-level `tf.level` entries and a final merged `tf` carrying the
nomenclature map that display nodes and exporters interpret; tables and
plots round out the set, with an `any` wildcard for terminal nodes.

What makes this a *dataflow* system rather than a mere graph of calls is
that structure and data are fused: each node's cache signature is a
structural hash of its type, manifest version, effective parameters and its
inputs' signatures, computed in topological order so that any upstream
change invalidates every downstream result without hashing data. Path
parameters contribute a stat signature — data-root-relative path, size,
modification time — in Nextflow fashion, so externally edited inputs
recompute. Editing a parameter, rewiring an edge, or touching a file
therefore reruns exactly the affected subgraph and nothing more.

This fusion of declaration and derivation — manifests describing structure,
handles carrying physics, signatures propagating dependency — lets the same
document serve the interactive browser, the REST API, and headless batch
execution alike, with identical semantics everywhere.
