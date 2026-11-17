# Randomized Gaussian-elimination search

## Detector/observable problem

The core input consists of a binary detector matrix (D), an observable matrix
(O), and a weight function. It searches for

\[
\min \{\operatorname{wt}(e):De^T=0,\ Oe^T\ne0\}.
\]

For a CSS quotient `ker(H_X) / row(H_Z)`, the detector matrix is `H_X`.
Nontriviality is tested against a basis of `ker(H_Z)`, because a vector is in
`row(H_Z)` exactly when it is orthogonal to all of `ker(H_Z)`.

For a general subsystem code with gauge space (G) and center (S=Z(G)), the
input symplectic rows are converted to ordinary detector/observable matrices by
swapping their X and Z blocks:

\[
D=S\Omega,\qquad O=L_{\rm bare}\Omega.
\]

Then `D e = 0` means (e\in S^\perp), while `O e != 0` means
(e\notin G). This searches the dressed quotient (S^\perp/G), including all
non-CSS cancellation sectors.

For a measurement operation, let \(G_{\rm aux}\) omit the deliberately
measured closed products \(C\), and let \(S_{\rm aux}=Z(G_{\rm aux})\). The
operation distance is

\[
\min\{\operatorname{wt}(e):e\in S_{\rm aux}^\perp
\setminus (G_{\rm aux}+C)\}.
\]

This uses \(S_{\rm aux}\Omega\) as the detector matrix and a basis of
\((G_{\rm aux}+C)^\perp/S_{\rm aux}\) as the observable Pauli rows. Keeping the
auxiliary centre as the detector space is essential: it retains errors that
flip a measured product, while quotienting by \(C\) declares the measured
products themselves benign.

## One trial

1. Compute a basis (K) of `ker(D)` once before the randomized loop.
2. Randomly order the physical columns.
3. Row-reduce (K) using that column order as the pivot order.
4. Test every reduced row for nonzero observable effect.
5. Retain the lowest-weight valid row.

This is the random information-set methodology used by QDistRnd. CSS trials
permute individual columns. General symplectic searches provide five exact
coordinate strategies:

- `grouped` permutes physical-qubit groups and exchanges the X/Z pivot order.
- `local-clifford` additionally samples one of all six one-qubit symplectic
  maps independently on every qubit before elimination.
- `location-linear` samples an arbitrary invertible binary coordinate map
  independently inside each declared fault location. This is `GL(2,2)` on a
  singleton and `GL(4,2)` on a two-qubit location. It preserves whether each
  location is zero and therefore preserves the exact grouped cost, while
  exposing a much larger family of pivot coordinates.
- `three-block` embeds `(x,z)` as `(x,z,x+z)`. Each nonidentity Pauli then has
  binary weight two, and a one-qubit Clifford is an `S_3` permutation of the
  three columns belonging to that qubit.
- `three-block-global` permits a global permutation of all redundant columns.
  It is retained as an experimental diagnostic and is not the default.

For the transformed methods, every candidate is projected back into the
original `X | Z` coordinates before its weight and logical effect are tested.
The three-block kernel is the image of the original detector kernel; the code
never takes an unrestricted kernel in the redundant space, which would admit
spurious odd-parity triplets.

When both columns of one physical qubit are pivots, the implementation also
tests the sum of the two corresponding pseudo-generator rows. An optional
`pair_window` tests pair sums among the lowest-weight reduced rows.

## Grouped fault locations

An optional location partition replaces physical Pauli weight by

\[
w_{\cal L}(e)=\left|\{L\in\mathcal L:\operatorname{supp}(e)\cap L\ne
\varnothing\}\right|.
\]

The Python API accepts the nontrivial one- and two-qubit groups explicitly and
completes the partition with singleton groups for every omitted qubit. Groups
must be disjoint. Thus any of the 15 nonidentity Pauli outcomes on a declared
two-qubit location has cost one, while a boundary data error has its usual
singleton cost.

This cost is used inside each trial: reduced rows, local pivot-row combinations,
and the optional pair window are all ranked with grouped cost. For a two-qubit
location the local search checks every nontrivial combination of the available
pivot rows belonging to either qubit, including all X/Z and redundant
three-block columns. The final witness is projected to the original `X | Z`
coordinates and rescored directly from the declared partition.

The optional `multi_qubit_group_order_weight` changes the information-set
distribution without changing the quotient or cost. Location groups are drawn
in a Plackett-Luce order using an exponential race: singleton rate is one and
two-qubit-group rate is the supplied positive weight. A value above one moves
paired locations earlier on average; below one moves them later. Columns within
each selected group remain randomly ordered. Because pivot order can help or
hurt depending on the matrix, this parameter must be calibrated on planted
challenges and is exactly `1.0` by default. The limiting value `0.0` uses a
stratified order with every shuffled singleton group before every shuffled
two-qubit group.

## Parallel reproducibility

Trial (i) receives a seed obtained by separately mixing the namespace and trial
index, then mixing their XOR. Therefore adjacent namespaces do not merely
reorder the same power-of-two trial-key set, and sampled information sets do
not depend on thread scheduling.
Equal-weight witnesses are resolved lexicographically, making fixed-budget
results reproducible across thread counts. Fail-fast searches execute bounded
parallel batches, so the stopping trial count may depend on batch size.

## Guarantees

Every returned witness is directly verifiable and gives an upper bound on
distance. Failure to find a lower-weight witness is not a distance proof.

The default `grouped` strategy currently gives the best throughput on the
frozen corpus. The other strategies are exposed because their hit probability
per information set can differ, especially for mixed-Pauli subsystem codes.

The implementation follows the published randomized information-set method,
not source copied from QDistRnd:

- L. P. Pryadko, V. A. Shabashov, and V. K. Kozin, "QDistRnd: A GAP package
  for computing the distance of quantum error-correcting codes", JOSS 7, 4120
  (2022).
- M. Webster et al., "Distance-Finding Algorithms for Quantum Codes and
  Circuits", arXiv:2603.22532 (2026).
