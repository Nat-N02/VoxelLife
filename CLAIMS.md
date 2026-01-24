SPI: overlap fraction of top-2% repair voxels between prints

RGI: P(repair∣high flux)/P(repair)

RLI: repair localization via normalized entropy

f_flux: fraction of perpendicular directions passing flux gate

FTP: overlap of top-2 outgoing directions per voxel across prints (or your definition)

Observed Properties:

1) The system produces persistent, spatially localized domains (PIDs)
 
SPI ≈ 0.05–0.11 (nonzero, stable over 10k ticks)
f_flux ≈ 0.009–0.013 (≈1% of perpendicular directions pass gate)
RLI ≈ 0.41–0.45 (repair concentrated, not uniform)
RGI ≈ 5–7 (repair strongly biased to high-flux voxels)

Dead world:

SPI = 0
RGI = 0
Repair_events = 0

Crystal world:

SPI ≈ near-zero
RGI ≈ ~2
f_flux ≈ ~0.07

2) Spatially localized

f_flux ≪ 1
RLI < 1
RGI ≫ 1

3) Phase transitions occur
Dead world (no repair, uniform degradation)
Crystal world (global ordered, high repair footprint)
PID world (localized persistent domains)

Numerically separable by:

SPI
RGI
f_flux
RLI
repair_events

4) Uniform degradation regime exists
repair_events = 0
RLI = 1.0
SPI = 0
RGI = 0
D_sum increases monotonically

5) Crystallized pattern regime exists
RGI ≈ ~2
SPI ≈ ~0–0.07
RLI ≈ ~0.31

6) Persistent domain regime exists
SPI > 0
RGI ≫ 1
f_flux ≪ 1

7) τ is a major control parameter that separates regimes under fixed secondary parameters.
Dead world and crystal world both show:

SPI ≈ 0
RGI low
repair_events = 0 or globally distributed

8) High-damage lines 
BSI ≈ ~1.02
D_q95 ≈ D_q99
VarD elevated and stable
sharp spatial gradients aligned with boundaries

9) Isolated low-damage regions
PID world:
D_q95 ≈ ~15.7
D_sum moderate
BSI ≈ ~1.03–1.04
SPI > 0

10) Global high-damage regime

Dead world:
D_sum ~75,000+
D_q95 ~20
repair_events = 0
RLI = 1

11) Not static patterns
SPI < 1
FTP ≈ ~0.36
RFC fluctuates

12) Not equilibria
sustained mean_flux > 0
sustained repair_events > 0 (except dead)
bounded but nonzero variance

13) Persistent structures act as attractors in macroscopic state space
The system converges to and remains within a bounded region of macroscopic state space characterized by stable values of (SPI, RGI, f_flux, RLI, E_sum, mean_flux).

14) PIDs mostly occur near regime boundaries 
Mapping mean SPI over the (τ,Wdecay) plane reveals a narrow, continuous ridge of high persistence separating uniform degradation and crystallized regimes. This indicates that persistent domains emerge preferentially near a phase boundary controlled primarily by the repair threshold τ, and remain robust across a broad range of secondary parameters.

15) Robust to secondary params 
Over a wide range of W_decay, SPI (and therefore PIDs) persistently stays high.

16) PID persistence is sustained by coupling between high transport activity and localized repair
Decoupling flux from repair by treating flux as the global average flux prevented repair, leading to SPI = 0.
Randomizing the flux that repair depends on to different voxels leads to no PIDs. Adding a radius to randomization makes PID-like attractors which have SPI = 0.

17) Persistent domains exhibit low-damage interiors relative to their surroundings, associated with stable damage gradients
D_in​ <D_b​ <D_out​
Gb​ increases over time while VarD increases system-wide

18) PIDs persist for O(10⁵–10⁶) ticks under stable conditions
PID_size, SPI, and centroid remain bounded and nonzero from tick ~5000 to >200000.

19) Some PIDs exhibit coherent mobility across the lattice
∥Δc(t)∥ continuous and bounded over thousands of ticks, moving but not jumping

20) Boundary concentration 
D_in​ <D_b​ <D_out​
Damage is lowest inside, highest outside, and the steepest gradient is at the boundary and that gradient strengthens over time.

21) Repair gating dominates behavior
Randomized flux → no PIDs
Global flux → no PIDs
Radius scrambling → non-PID attractors

22) Robust to moderate changes in grid size and aspect ratio within tested ranges →  target
128x48x3 and 48x48x3 only have slight differences in SPI

23) At low τ, repair synchronizes globally, producing a lattice-wide oscillatory state that suppresses spatial differentiation and collapses into a low-energy attractor
tick 241: D_sum = 5.190
tick 242: D_sum = 8.425
tick 243: D_sum = 5.193
tick 244: D_sum = 8.428
tick 245: D_sum = 5.196
at τ = 0.3
repair events collapse down to 0 eventually. 0 SPI

24) If system rules are violated to allow measured flux to be from a random nearby voxel, coreless PIDs form.
0 SPI with low damage structures

25) Boundary collapse under routing erosion
PIDs expand to a superstructure with rising W_decay. This leads to repair localizing and non-repaired areas collapsing, leading to expansion into those areas, resulting in more repair localization. RGI is roughly 3, much lower than usual. SPI > 0.05, indicating little structure.

26) Damage exposure causes fission or collapse
When a high-damage perturbation severs the transport envelope of a PID without fully saturating the surrounding lattice, the domain undergoes fission into two or more persistent subdomains rather than collapsing.

27) In the crystallized regime, spatially fixed, globally coupled transport patterns emerge
Correlation between dominant axis in a voxel and the voxel next to it rises dramatically in crystallized structures.

28) PIDs exhibit partial recovery following perturbations, re-establishing internal structure and transport pathways
After perturbing PIDs with global damage increase, weak connections between PIDs fall apart and PIDs lose outer layers. +0.1 damage was added for 100 ticks. Internal FTP fell by 0.1, but recovered.
perturbation 16k–17k, recovery 17k–23k