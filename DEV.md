[1] Introduced a voxel-based dynamical system 

Lattice: L={0..Nx−1}×{0..Ny−1}×{0..Nz−1}
State per voxel: si,j,k(t)=(E,D,P,R,W)
Global state: S(t)={si,j,k(t)}(i,j,k)∈L
Update rule: S(t+1)=F(S(t);θ)
Time: integer ticks t∈N

Two runs with same seed lead to same result (at least, on the same system)
| Formal Object | Code Representation        | Evidence               |
| ------------- | -------------------------- | ---------------------- |
| Lattice (L)   | `nx, ny, nz`, `idx(x,y,z)` | `idx` bijection test   |
| State (E)     | `curr.E[i]`                | length = nvox          |
| State (D)     | `curr.D[i]`                | length = nvox          |
| State (P)     | `curr.P[i]`                | length = nvox          |
| Weights (W_d) | `curr.W[i*6 + d]`          | sum normalized         |
| Time (t)      | `tick`                     | increments in `step()` |
| Update (F)    | `step()`                   | curr → next → swap     |

Proved that energy isn't routing to non-neighbors

“VRD is implemented as a discrete-time dynamical system defined on a three-dimensional rectilinear lattice. System state is fully specified by per-voxel variables 
(E,D,P,R,W)
(E,D,P,R,W), and global evolution proceeds by synchronous application of local update rules that depend only on nearest-neighbor interactions.”

[2] Includes local energy routing 
Routing turned off led to no energy movement.

[3] Includes damage accumulation 
Routing turned off led to no damage accumulation.

[4] Includes repair gating 
No flux gate -> collapse
if (cost > Ej) return; disabled, mass repair

[5] No explicit boundaries 
There is no state variable encoding region identity
There is no rule that tests for “membership” in a structure
All update rules are uniform across the lattice
Interactions are strictly local and neighbor-based

[6] No fitness functions 
VRD does not implement evolutionary fitness, selection, or reproduction; however, access to repair is governed by a strictly local, relative performance threshold based on transport activity.

[7] No replication mechanisms
No state is ever copied conditionally into another region
There is no “spawn,” “seed,” or “initialize from neighbor” rule
New domains arise only through continuous local dynamics, not discrete reproduction events

[8] VRD is 3D voxel-based 
There are three dimensions, and they are discrete. 

[9] Uses energy flow 
There exists a scalar energy field that is transported between voxels and dissipated through routing loss, leakage, and repair cost.
It is transferred between voxels
Transfer affects future state

[10] Uses damage accumulation 
Removing repair and annealing led to constant damage accumulation.
There exists a scalar energy field that is transported between voxels and dissipated through routing loss, leakage, and repair cost.

[11] No identity encoded 
Every voxel is only a tuple of continuous variables — no symbolic or categorical state exists

[12] No selection encoded 
All thresholds act on fields, not entities

[13] Has Nx × Ny × Nz
nvox == nx * ny * nz
idx(x,y,z) is bijective
Neighbor table size = nvox * 6

[14] Integer coordinates 
inline int idx(int x, int y, int z)
All coordinates are int, all access uses integer arithmetic.

[15] Has E
std::vector<float> E;

[16] Has D
std::vector<float> D;

[17] Has P
std::vector<float> P;

[18] Has R_boost
std::vector<float> R_boost;

[19] Has W ∈ ℝ⁶
std::vector<float> W; packed [i*6 + d]

[20] Updated each tick
void step()
...
tick++;

Only one tick increment exists, and all state transitions go through step().

[21] Only nearest neighbors
nbr[i*6 + XP]
nbr[i*6 + XN]
nbr[i*6 + YP]
nbr[i*6 + YN]
nbr[i*6 + ZP]
nbr[i*6 + ZN]


Built exclusively in:

build_neighbors_and_hashes()

[22] Renormalized weights
normalize6(w);

[23] Weights sum to 1
float inv = 1.0f / sum;
w[d] *= inv;

[24] Default grid size exists
Params {
    int nx=128, ny=128, nz=4;
}

[25] Dominant axis definition 
inline AxisInfo dominant_axis(size_t i) = max(|W_x|, |W_y|, |W_z|)

[26] Perpendicular neighbor checks
Only 4 neighbors are selected, excluding dominant axis:
if (ax.axis == XP) { Y, Z }

[27] Threshold τ exists 
float eff_thresh = tail_frac * sent_tail_local[i];

[28] τ defined as flux fraction
if (sent[j] < eff_thresh) return;

[29] Uniform plane injection at z=1
inline bool is_source_voxel(int x,int y,int z) const {
    return (z == 1);
}

[30] No locomotion rules
There is no function that modifies (x,y,z)
Only E, D, P, W, R evolve. Position is immutable.

[31] Metrics are aggregate
All metrics operate on:
std::vector<float> curr.x (where x is a given value in curr)
std::vector<float> sent
and reduce to scalars:
double BSI, RFC, SPI, FTP
None feed back into step().

[32] From local rules alone
step() reads only:
curr.[i]
nbr[i*6 + d]
and the local sent_tail_max.

[33] Not SOC 
There is no:
Slow global drive
Fast avalanche mechanism
Threshold-triggered cascades
There exists a scalar energy field that is transported between voxels and dissipated through routing loss, leakage, and repair cost.

[34] Baseline energy loss
float eff_leak = p.E_global_leak;

[35] Damage-proportional loss
eff_leak = p.E_global_leak * (1.0f + p.leak_k * x * x);

[36] R reduces loss
float c = conductivity(curr.D[i]) * (1.0f + R_boost[i]);

[37] Repair increases R
R_boost[i] += dR;

[38] P replenishes
next.P[i] += alpha_eff * (p.P0 - curr.P[i]);

[39] P consumed
next.P[i] -= dR;

[40] R increase requires P
dR = std::min(dR, next.P[i]);

[41] W decays passively
w[d] = lerpf(cw[d], 1.0f / 6.0f, p.W_decay);

[42] Damage accelerates decay
float noise_amp = p.D_to_noise * D + p.porous_noise_boost * x * x;

[43] Analysis is qualitative
All indices: BSI, SPI, FTP, RFC, RLI
are descriptive, not used in dynamics.

[44] Energy routed by W
sent_dir[i*6 + d] = send * frac;
next.E[j] += delivered * frac;

[45] Sum ≤ 1
Normalization enforces it:
normalize6(w);

[46] Local neighborhood radius 
The radius is p.sent_tail_radius
and the neighborhood is explicitly sliding_max_1d_padded(..., r, ...).

[47] Figure reflects geometry 
The figure is a direct schematic of the neighbor topology and dominant-axis logic implemented in build_neighbors_and_hashes() and apply_perpendicular_repair().

[48] Depends on flux 
Damage: p.D_activity_gain * sent[i]
Repair: float a_norm = sent[j] / (curr.E[j] + 1e-6f);

If sent = 0, then:
No damage accumulation
No repair
No R generation

[49] Energy fraction gate 
float eff_thresh = tail_frac * sent_tail_local[i];
if (sent[j] < eff_thresh) return;

[50] Energy cost gate 
float Ej = E_residual[j];
if (cost > Ej) return;

[51] R increases efficiency 
float c = conductivity(D) * (1.0f + R_boost[i]);
float send = std::min(Ei * c, Ei);

[52] VRD is dissipative 
In the absence of source terms, the system exhibits a strictly negative energy balance.

OTHER NOTES:

Synchronous update:
All state transitions are computed from S(t) and committed simultaneously to form S(t+1)
Deterministic evolution:
Given identical (S(0), θ, seed), VRD produces identical S(t) for all t.
Continuous state on discrete lattice:
All per-voxel state variables are real-valued scalars evolving under continuous update rules on a discrete spatial grid.
Energy is not conserved:
Transport, leakage, and repair introduce explicit sinks that remove energy from the system each tick.
Spatially homogeneous rules:
All update rules and parameters are globally uniform across the lattice.
Externally driven system:
VRD includes an exogenous energy source term applied independently of internal state.
Transport capacity is state-dependent:
The fraction of energy a voxel can route is a function of its local damage and repair state.
Finite-range coupling:
All direct interactions are limited to nearest neighbors (radius-1 stencil), with no long-range state access.
Routing weights lie on a 5-simplex:
For each voxel, outgoing routing weights form a normalized probability distribution over six directions.
Static lattice topology:
The neighborhood graph is fixed for the duration of the simulation.
Phase-ordered update pipeline:
Each tick consists of a fixed sequence of transport, repair, weight evolution, source injection, and state commit stages.
Topology-aware routing:
Boundary voxels retain six-direction weight vectors, but transport is only delivered to in-bounds neighbors. Weight renormalization causes the routing distribution to adapt to lattice topology.