# Statistical methods

What each analysis step computes, why, what it assumes, and how to read every
number it produces. The [README](../README.md) covers how to run the pipeline;
[REFERENCE.md](REFERENCE.md) lists every setting and file.

The running example is the validation experiment this pipeline was first tested
on: three hPSC lines (iPSC465, iPSC522, ESI017), BMP4 for 48 h on 700 µm
micropatterns, stained DAPI / ISL1 / TBXT / SOX2, five colonies per line, one
well per line. Nothing below is specific to it.

---

## The statistics in plain words

Read this first if you do not work with statistics every day. Each idea below
is used somewhere in the pipeline; the numbered sections that follow give the
exact recipe.

**Replicate (what counts as n).** A replicate is one independent repeat of the
experiment's comparison. A colony with 2,000 nuclei is still *one* colony: its
nuclei grew together, were stained together and were imaged together, so they
are not 2,000 independent tries. The pipeline therefore uses nuclei only to
measure each colony's profile precisely, and uses colonies (or wells, or
independent differentiations) as n. Counting nuclei as n is called
*pseudoreplication*; it makes tiny, meaningless differences look
"significant".

**Confidence interval (CI).** The shaded band around a mean curve. A 95 % CI is
a range built so that, if you repeated the experiment many times, 95 % of such
ranges would contain the true mean. Wide band = the replicates disagree, or
there are few of them. Bands around a single colony's own curve only say how
well that colony is measured; they are not for comparing conditions.

**p-value.** How surprising the data would be if the conditions truly did not
differ. p = 0.01 means that a difference at least this large would turn up in
about 1 of 100 experiments where nothing is going on. It is not the probability
that the result is wrong, and it says nothing about how *large* the effect is
(look at the curves and the effect sizes for that).

**Permutation test (how every p-value here is made).** Write each colony's
condition on a card, shuffle the cards, deal them back out and recompute the
difference between conditions. Do that thousands of times. If the conditions
do not matter, the real labeling is just one of these shuffles, so the real
difference should look ordinary among the shuffled ones. The p-value is the
fraction of shuffles that give a difference at least as large as the real one.
Nothing is assumed about bell curves or equal spreads.

**The smallest possible p.** With few replicates there are only so many
shuffles. Five colonies against five can be shuffled 252 ways, so p can never go
below about 0.008, and 3 wells against 3 can never reach 0.05, however strong
the effect. The pipeline lists this floor next to every test. More replicates,
not more nuclei, are what lower it.

**Testing a whole curve, and finding where it differs.** A radial profile is a
curve, so the pipeline tests the whole curve at once (one global p) instead of
running dozens of tests, one per distance bin, and getting some "hits" by
chance. To say *where* along the radius the conditions differ, each shuffle also
records its largest difference anywhere on the curve; a distance counts as
different only if the real data beat 95 % of those shuffled maxima. That is the
gray bar under the curves. It keeps the chance of even one false bar at 5 %.

**q-value (false discovery rate).** When many pairs or markers are tested, some
p-values come out small by luck. A q-value is a p-value adjusted for how many
tests were run (the Benjamini-Hochberg method): calling everything with
q < 0.05 "different" means that, on average, at most 5 % of those calls are
false.

**Bootstrap (how a single colony's numbers get error bars).** You cannot image
a colony again, but you can ask how much a number (a boundary position, a peak)
would wobble if the colony had looked slightly different. The bootstrap answers
that by resampling: draw pieces of the colony at random, *with replacement* (so
some pieces appear twice and some not at all), recompute the number, and repeat
200 times. The spread of the 200 answers is the number's standard error (SE).
Here the pieces are angular sectors of the colony, like slices of a pie, rather
than single nuclei: neighboring nuclei are alike, so resampling nuclei one at a
time would make the colony look more certain than it is. A boundary that is
ragged around the colony gets a large SE, because different slices disagree.
A number is reported only if at least 80 % of the resampled colonies find it.

**Bootstrap for a trend.** For "does the boundary move with dose?", replicates
are resampled within each dose and the slope refitted each time; the middle
95 % of the refitted slopes is the slope's CI.

**Fitting a boundary (logistic fit).** Across a domain border a marker goes
from on to off. The pipeline fits an S-shaped curve to that switch. Its
midpoint (x50) is where the marker is half on, the boundary position; the
distance over which it goes from 10 % to 90 % is the width, how sharp the border
is. Both are in µm and do not depend on how bright the stain was.

**Positivity gate.** The intensity above which a nucleus is called positive:
the level that only 1 % of nuclei in the negative-control wells exceed. Without
a control, the pipeline looks for two clearly separate populations (bright and
dim) with a mixture model; if it does not find them, the marker gets no gate
rather than a guess.

**Robust z and MAD (QC flags).** To find an odd colony, each colony is compared
with the median of its group, in units of the typical spread (the median
absolute deviation, MAD). Medians are used because one bad colony cannot drag
them, unlike a mean and standard deviation.

**Variance components.** Colonies differ from one another, wells differ a bit
more, separate differentiations differ more still. The pipeline estimates how
much of the spread comes from each level. That tells you where extra effort pays:
if most variation is between differentiations, imaging more colonies will not
help; doing another differentiation will.

**Power.** The chance that an experiment of a given size detects an effect of a
given size. From the measured variances, the pipeline works out the smallest
difference that 2, 3, 4 or 6 independent experiments could reliably detect, so
you can plan the next experiment before doing it.

---

## 0. The hierarchy, and the unit of replication

```
condition (cell line, dose, …)  ─►  well  ─►  colony  ─►  nucleus
                                              (replicate)   (measurement; ~2 000-3 000 per colony)
```

Nuclei in one colony are not independent: they share a colony, a staining
reaction, an imaging field and, because fate is organized radially, strong
spatial correlation. Treating thousands of nuclei as independent samples
(pseudoreplication) would shrink standard errors roughly √2000-fold and make
everything "significant".

So **nuclei estimate a colony's profile, and colonies (or wells) estimate and test
the condition effect.** Every between-condition interval and p-value in the
pipeline has n = colonies or n = wells, never nuclei (§5).

---

## 1. From pixels to a value per nucleus

**Segmentation.** Cellpose-SAM, in 2-D on the DAPI maximum projection or in 3-D
on the z-stack; masks are size-filtered on a log scale to drop debris and merges.

**Per-nucleus intensity.** The median of each marker's pixels inside the nuclear
mask: robust to hot pixels and to bleed at the nuclear boundary.

**DAPI normalization.** By default each marker value is divided by that nucleus's
DAPI level (`default_space: dapi_norm`). This corrects the dominant nuisance in
projected confocal data: how much nuclear material sits in the optical section.
Non-nuclear markers can stay raw (`markers: {X: {space: raw}}`), and every
marker should when the nuclear channel is not a DNA stain (an H2B fusion varies
from cell to cell, so dividing by it adds variance instead of removing it):
`default_space: raw`. Signaling markers have their own values (§10c).

**Which nuclear stain.** `nuclear_stain: auto` reads the nuclear channel's name
(`H2B`, `H3`, `histone` → histone; anything else → DNA). When a marker is
divided by a histone label the run warns and every caption says so: histone
levels change with chromatin compaction and cell cycle, so the ratio is no
longer a correction for optical section.

**Saturated nuclei.** A nucleus whose median pixel is at the detector ceiling in
any channel is removed before anything else is computed (§10d, *Saturated
nuclei*).

---

## 2. Colony geometry and the radial coordinate

### 2.1 The main colony

The nuclear masks are joined into a footprint by a morphological closing with a
disk of radius `close_radius_um` (about one to two nuclei; it bridges gaps
between nuclei but not between colonies), and holes are filled. The largest
connected component is the colony of interest; other components larger than
`min_colony_area_um2` are neighboring colonies that crept into the field.
Nuclei outside the main colony are excluded from every profile.

### 2.2 The pattern circle

A circle is fitted to the colony's rim: only where the colony meets background,
never where it meets the image border. The fit is algebraic (Kåsa) with iterative
trimming: rim points whose distance from the current circle deviates from the
median by more than 2.5 robust SDs are set aside and the circle refitted until
the kept set is stable, then refined by Gauss-Newton on the geometric distance.
This locks onto the dominant arc when part of the rim is a tail of escaped cells.
The mask is then cut at 1.1 × R (`trim_outside_circle`): cells that left the
pattern are not part of the colony whose pattern is being measured.

### 2.3 Distance from the edge (the primary coordinate)

Micropattern fates are organized relative to the **edge**, where BMP signaling
is received, not relative to the center. The distance of a nucleus from the edge
is the Euclidean distance transform of the colony mask at its centroid, converted
to µm with the image's pixel size. Measured on the mask, it follows the colony's
real outline rather than an idealized circle.

A colony cut by the image border has no visible edge on that side, and the
distance transform would measure nuclei there to the far edge. When more than
`max_border_fraction` of the rim lies on the border, the distance is taken from
the fitted circle (`R − r`) instead, and the colony is flagged `clipped`. On
synthetic colonies with known geometry the mask method recovers edge distances
to a median of −2 µm; for a colony with 27 % of its rim cut off, the circle
method stays within 15 µm (95th percentile) where the mask alone is off by more
than 30 µm (`tests/test_micropattern.py`).

### 2.4 The normalized coordinate

`r_norm = dist_edge / R`, 0 at the edge and 1 at the center, with R the fitted
radius. It makes colonies of one nominal size directly comparable. It is **not**
used to compare colonies of different pattern sizes: if plate-map
`diameter_column` varies within a comparison, the normalized-grid tests are
skipped and only the µm grid is tested.

---

## 3. One intensity scale per scope

> **In plain words.** Every marker is put on a 0-to-1 scale, but one scale for all colonies stained and imaged together, never one per colony. So a colony with half as much protein reads half as high, and a colony with none stays near 0 instead of having its noise stretched into a fake pattern.

To put markers on a common, readable axis every value is rescaled linearly with
anchors taken **once for all colonies of the scope** (by default one plate ×
stain panel):

```
scaled = (value − P1) / (P99 − P1)
```

with P1 and P99 the 1st and 99th percentiles of all in-colony nuclei of the
scope (kept colonies only). Because the anchors are shared:

* **Levels are kept.** A line expressing ISL1 at half the level of another reads
  half as high; the tests can detect it. On synthetic data a half-amplitude
  colony reads 0.49 × as high.
* **No fake patterns.** A colony that does not express a marker stays near 0.
  Rescaling each colony to its own range would stretch its noise into a full-height
  "pattern".

The transform is linear and unclipped, so means of scaled values are scaled
means and nothing is truncated.

**The assumption** is that colonies on one scale were stained and imaged alike
(one antibody mix, one exposure, one plate). That is the same assumption a
positivity threshold estimated over the scope makes. A well stained more
brightly than its neighbors would read as higher expression; the per-marker QC
(§9) compares colonies within a well to catch local staining artifacts, but a
whole-well difference is indistinguishable from biology when each condition is a
single well (§5).

---

## 4. Per-colony profiles

### 4.1 Two grids

* **µm grid**: bins of `bin_um` (default 10 µm, about one nucleus) from the edge
  inward. Each colony's curve ends at its own radius.
* **Normalized grid**: `n_norm_bins` (default 25) equal bins of `r_norm`.

A colony's value in a bin is the mean scaled intensity of its nuclei there, plus
the fraction of those nuclei that are positive (§7). A bin with fewer than
`min_nuclei_per_bin` nuclei is left empty: near the center a disk has little area
and few nuclei, and a mean of two nuclei is noise.

### 4.2 Per-colony figures

For looking at one colony, its profile is also drawn on its own bins, in µm from
the edge:

* `quantile` (default): bin edges at quantiles of the nuclei's distances, so each
  bin holds about the same number of nuclei: stable per-bin estimates.
* `equal_area`: bin edges at quantiles of the colony mask's pixel distances, so
  each bin covers about the same colony area: for a disk, equal-area annuli.
* `fixed`: `bin_um`-wide bins.

The band is a t-based 95 % CI over the nuclei in each bin (`ci_method:
bootstrap` for a percentile bootstrap). It says how precisely the colony's own
curve is known. **It is not an error bar for comparing conditions**: nuclei are
not replicates.

---

## 5. From colonies to wells and conditions

> **In plain words.** First decide what counts as n (see the primer above): with three or more wells per condition the pipeline uses wells, otherwise colonies, and the report says which and what that means. Then average the replicate curves and draw a 95 % CI band.

### 5.1 The replication unit

Colonies in one well share one seeding, one dose and one staining reaction, so
they are not fully independent of each other. `stats.replicate: auto` therefore
chooses:

| Design                        | Replicate                                           | Consequence, stated in the report                                                                                           |
| ----------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| every condition has ≥ 3 wells | **well** (colonies averaged within each well first) | inference generalizes to new wells                                                                                          |
| a condition has 2 wells       | colony                                              | colonies of one well are partly dependent; p-values are optimistic                                                          |
| a condition has 1 well        | colony                                              | well and condition are confounded: a difference shows that the colonies differ, not that it would replicate in another well |

The example experiment is the third case. Its conclusions are about these
colonies; replicate wells, ideally from independent staining runs, are what turn
them into claims about the lines.

**Independent experiments.** Wells of one differentiation share its batch, so
they are not independent either. Name the plate-map column that identifies
differentiations in `stats.experiment_column`; when every condition has at least
`stats.min_experiments_for_unit` (2) of them, the experiment is the replicate:
colony curves are averaged per well, wells per experiment (each well weighs the
same), and n = experiments. Two differentiations are then never pooled into one
curve; each contributes one replicate to the condition mean.

### 5.2 Means and intervals

At each grid point the well or condition curve is the mean across replicate
curves, with SEM and a t-based 95 % CI (n = replicates). A point covered by fewer
than `min_replicates` replicates (default 3) is left empty instead of drawing a
spike with a huge interval. Individual colony curves are drawn behind the means
in the *every colony, by well* galleries, so colony-to-colony reproducibility is
visible directly.

---

## 6. Comparing conditions: whole curves

> **In plain words.** One shuffle test on the whole curve answers "do the conditions differ?" (the global p in each figure title). A second use of the same shuffles answers "where along the radius?" (the gray bar). See *Permutation test* and *Testing a whole curve* in the primer.

A radial profile is a curve, not a number. The right object is the function
f_i(d) of each replicate i, and the right question is whether the distributions
of these functions differ between conditions: functional data analysis.
Reducing a curve to one number first (the peak, an average) throws away where
along the radius the difference lies.

### 6.1 The point-wise statistic

At each grid point the one-way ANOVA F compares the condition means of the
replicate values to the scatter between replicates within a condition:

```
F(d) = between-condition mean square / within-condition mean square     at distance d
```

Only grid points where every condition has at least two replicates are used.

### 6.2 The global test

The global statistic is the mean F over the grid, T = mean_d F(d): the evidence
for a condition effect integrated along the radius. Its null distribution comes
from **permuting the replicate → condition labels**. Under the null hypothesis
that the condition carries no information about a replicate's profile, the labels
are exchangeable, so each relabeling is as likely as the observed one:

```
p = (#{T_perm ≥ T_obs} + 1) / (n_permutations + 1)
```

The test assumes no normality, no equal variances and no form for the curves:
only exchangeability of replicates under the null. The unit permuted is the
replicate (colony or well), which is what makes n correct.

**Exact enumeration.** When the number of distinct relabelings is no larger than
`n_permutations`, all of them are enumerated and p is exact. With 5 vs 5 colonies
there are 252 relabelings, half of them mirror images that give the same F, so
the smallest possible p is 2/252 ≈ 0.008. With 2 conditions × 3 wells it is
2/20 = 0.1, and the test cannot reach 0.05 however large the effect. The
smallest possible p is reported next to every test
(`compare_global.csv: smallest_possible_p`). It is a property of the design,
not of the data: **the number of replicates, not the number of nuclei, sets the
resolution.**

### 6.3 Where along the radius: max-F control

To say *where* conditions differ without declaring significance at dozens of
correlated grid points by chance, each permutation records the **maximum** F
over the grid. The 95th percentile of that maximum is one threshold for the whole
curve; grid points whose observed F exceeds it are significant with the
family-wise error rate controlled at 5 % (the max-statistic method of
Westfall-Young and of Nichols & Holmes). They are drawn as a gray bar under the
curves and listed as `significant_from` / `significant_to` in µm or `r_norm`.

**Reading it.** The global p answers *do the conditions differ?*; the bar answers
*over which band of the colony?* In the example, SOX2 differs from about 80 µm
inward (the center domain), ISL1 between 50 and 180 µm (the inner flank of the
edge domain), not in the outermost 50 µm, where colony-to-colony scatter is
too large for the test to separate the lines.

### 6.4 Validation

On synthetic curves with no difference, the global test rejects at 3.5 % for
α = 0.05 (200 simulations); a difference confined to the outer quarter of the
radius is found (p = 0.001) and localized there, with no significant point in
the inner half (`tests/test_micropattern.py`).

---

## 7. Pairwise tests

> **In plain words.** With three or more conditions, every pair is tested the same way and the p-values are turned into q-values so that testing many pairs does not produce false hits.

A significant global test says *some* conditions differ. With three or more
conditions, the same functional test is run on every pair, restricted to those
two conditions' replicates. The p-values of one family (one grid and one
readout, across all markers and pairs) are adjusted by the Benjamini-Hochberg
procedure to q-values, controlling the expected proportion of false discoveries
among the pairs called different (`q < 0.05`). The heat map
(`figures/compare/pairwise_*.png`) shows every q.

---

## 8. Interpretable features

Each colony's curve is also reduced to scalars that are easy to communicate:

| Feature         | Definition                                                                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `edge_value`    | mean level in the outermost `edge_band_um` (default 50 µm)                                                                                                                          |
| `center_value`  | mean level at r_norm ≥ 0.85 (the inner 15 % of the radius)                                                                                                                          |
| `mean_value`    | mean level over the whole µm profile                                                                                                                                                |
| `peak_um`       | distance from the edge of the profile's maximum                                                                                                                                     |
| `transition_um` | first distance from the edge where the profile crosses halfway between its edge and center levels: the width of an edge domain (ISL1, CDX2), or where a center domain (SOX2) begins |

Features are averaged within each replicate and compared across conditions with
the Kruskal-Wallis test (rank-based, robust to small n and outliers), with BH
q-values across markers × features. With five replicates per group the
Kruskal-Wallis test has little power; treat the features as interpretation and
cross-checks of the functional test, not as the primary analysis.

---

## 9. Similarity of colonies

Independently of any grouping, every pair of colonies gets a distance: the
root-mean-square difference between their concatenated normalized-grid curves
(all markers, shared scale), over the grid points both cover. The heat map
orders colonies by condition and well. If colonies of one condition are more
alike than colonies of different conditions, blocks appear on the diagonal
without the analysis having been told the labels: assumption-light evidence
that the replicates are coherent and the condition effect real. A colony that is
far from its own group is worth opening in the overlays. Mean within- and
between-condition distances are tabulated in `similarity_within_between.csv`.

---

## 10. % positive profiles

> **In plain words.** The fraction of nuclei brighter than a gate set on the negative control, at each distance from the edge. No control and no clear bright/dim split means no % positive curve, never an arbitrary cutoff.

The second readout is the fraction of nuclei above the marker's positivity gate
in each bin, computed and compared exactly like the intensity (§4-§7). The gate
is the `1 − fpr` quantile (default the 99th percentile) of the marker in its
negative-control wells, so at most 1 % of true negatives are called positive. A
marker's negative control can be a different well from other markers' (a
pluripotent well is negative for TBXT but positive for SOX2; declare it with
`positive_for` / `control_for` in the plate map). Without a control, a
two-component mixture is fitted and accepted only when the two populations are
clearly separated; otherwise the marker gets no gate and no % positive profile,
rather than an arbitrary one. In the example there are no control wells, so all
three gates come from the mixture fit; the threshold evidence panel shows where
each gate sits on its histogram.

---

## 10b. Several markers: local trends and co-expression

**Local trends.** The stacked multi-marker figures add, for each marker and
position, `z = (m[i+k] − m[i−k]) / sqrt(se[i+k]² + se[i−k]²)`: the change across
the neighboring bins in units of its standard error (over nuclei for one colony,
over colonies for a well). Dividing by each marker's own noise puts markers with
very different ranges on one scale without rescaling any colony. The z-values
are descriptive: nothing is corrected for the number of positions and markers,
so a single marked cell is a pointer to look at the curve, not a finding.

**Co-expression against distance.** For two markers measured in the same
nuclei, each colony's nuclei are binned by distance from the edge and, per bin,
the Spearman (or Pearson) correlation of the two shared-scale values is
computed, with a Fisher-z interval (Spearman SE `sqrt(1.06 / (n − 3))`, Bonett
and Wright 2000). With gates, the double-positive fraction is compared with the
product of the two single-positive fractions, which is what it would be if the
markers were independent within the bin; their ratio (log2) is reported as an
enrichment. A within-bin correlation removes most of the shared radial trend, so
it measures whether *neighbors at the same distance* co-vary, not whether the two
profiles have the same shape (that is what the stacked profiles show). A
residual gradient inside a wide bin still contributes, which is why the bins are
kept narrow relative to the patterns (25 µm by default).

**Colonies first.** Across a well, each readout is computed per colony and the
colony values are averaged (correlations on the Fisher-z scale), with n =
colonies and the `min_replicates` rule of §5. Correlating pooled nuclei would
mix within-colony co-variation with between-colony differences in level and can
show a strong correlation that exists in no single colony (Simpson's paradox;
`tests/test_micropattern.py` checks that the pipeline reports ~0 where pooled
nuclei give 0.88).

---

## 10c. Signaling markers

A signaling reporter (β-catenin, SMAD2/3, SMAD4, ...; recognized by name, see
REFERENCE.md) carries its information in where the protein is, so it gets a
per-nucleus value of its own instead of a shared-scale intensity, and no gate.

**Background.** `b` = median of the cell-free pixels (farther than
`free_distance_um` from every nucleus) in a band `local_band_um` wide just
outside the cell-covered area, whose holes are filled: on a micropattern, the
passivated surface right around the colony. A median of a local band is robust
to debris (a minority of bright pixels), to vignetting (the band sits at the
colony, not in the image corners) and to unsegmented cells inside the colony,
which a whole-image cell-free mask would count as background. Fallbacks: all
cell-free pixels outside the covered area, then the camera offset (0.5th
percentile). An image whose `b` is far from the other images of the same
dataset and stain set (> 5 × 1.4826 MAD and > 10 %) is flagged in
`signaling_background.csv`. Stain sets are compared separately because a
different antibody in the same channel has a different background.

How much the choice matters is an empirical question worth checking once per
design: on `hd_micropatterns_bcat` the local and whole-image backgrounds differ
by 15-110 a.u. against nuclear β-catenin of 7 000-20 000 a.u. There the
background was never the problem; the reference was (below).

**Nuclear readout** (β-catenin): `N − b`, with `N` the median over the nucleus
core (the 1-pixel rim removed, which mixes nucleus and cytoplasm at any
realistic resolution). Not divided by the nuclear stain.

**N:C readout** (SMADs): `(N − b) / (C − b)`, with `C` the median over a ring
1 to 4 µm outside the nucleus. Rings grow until they meet a neighbor's, so no
pixel counts twice and no ring enters another nucleus; in 3-D they grow within
each z-plane. Ring pixels must lie in cell bodies (the smoothed channel above
`b + body_k · MAD` of the cell-free pixels): beyond the colony edge a ring would
otherwise reach the empty surface, pull `C` toward `b` and inflate N:C exactly
at the rim. A nucleus whose ring is too small or does not clear `b` has no N:C.

**Relative to a reference.** Per marker and scope (plate × panel by default),
`R` = the median over reference colonies (pluripotent by default) of each
colony's median value, and every value is reported as `value / R`, so 1 is the
reference state. Colony medians first, so one large colony does not outweigh
several small ones. A panel without reference colonies uses those of its plate;
a marker without any keeps its unnormalized values everywhere, so one marker
never mixes units. `R` rests on however many reference colonies there are:
`tables/signaling_reference.csv` lists the count, and with one colony the
reference carries that colony's noise into every value.

**No reference.** `reference_values: []` keeps the background-subtracted values
`N − b` in a.u. Use it when the reference wells cannot carry the
normalization: few colonies, sparse or under-segmented ones (look at their
nuclei counts and density), or a reference that is not the biological ground
state the question needs. Values are then comparable within one stain panel
(one staining reaction, one exposure) but not across panels.

From here on these values are treated exactly like any other: per-colony
curves, replicate means and the functional tests of §6.

---

## 10d. Boundaries and peaks

> **In plain words.** For each colony: where each marker switches off and how sharply (an S-curve fit), where two facing domains meet, and where a signaling peak sits. Each number gets an error bar from resampling slices of the colony (the bootstrap in the primer). Conditions are then compared on the spread of these numbers across colonies or wells, not on the per-colony error bars.

The profile is the object for testing; the claims are scalars. The `boundary`
step reduces each colony to a few of them, each with an uncertainty.

**Boundary of one marker.** Across its transition a marker is fitted with a
logistic, f(d) = 1 / (1 + exp(−k (d − x50))), d the distance from the edge:
a binomial fit to the per-nucleus gate calls (`readout = positive`, the fate
switch) and, for any marker, lo + (hi − lo) f(d) fitted to the binned mean on
the shared scale (`readout = intensity`). x50 is the boundary position; the
width is the 10 %-to-90 % distance, 2 ln 9 / |k|, also in units of the colony's
median nearest-neighbor nuclear spacing. The fit is restricted to the monotone
stretch between the smoothed profile's minimum and maximum, so a domain that sags
toward the center or bumps at the very edge does not widen the fit. A fit is
accepted when x50 lies inside the colony, the width is smaller than the colony
radius and, for intensity, the transition spans at least `min_amplitude` of the
shared scale; otherwise the colony reports no boundary for that marker (a flat
colony fits a tiny step anywhere).

Unlike peak height divided by FWHM, x50 and the width do not depend on staining
brightness, and they are measured on the transition rather than on the width of
the domain.

**Facing pairs.** Two markers of one stain set whose domains face each other (one
high at the edge, one high in the center; found automatically from the gate calls,
or listed in `boundary.pairs`) define a boundary where their fitted curves cross,
each on its own 0-1 range: (k_A x_A − k_B x_B) / (k_A − k_B). The overlap is x50
of the edge-high marker minus x50 of the center-high one: positive when a band of
nuclei is on for both, negative for a gap where neither is. The double-positive
fraction within `window_um` of the crossing is compared with the product of the
two marginal fractions there, which is what independent markers would give; a
mutually repressive pair falls far below it.

**Interior peak.** For signaling markers the profile is smoothed (kernel-weighted
mean) and the most prominent local maximum beyond `edge_exclude_um` is taken,
refined by a parabola through its neighbors; it must rise `min_peak_ratio` above
the minimum between it and the edge (the trough). Peak / trough is scale-free and
so comparable across stain panels. The offset of the peak from every boundary of
the same colony is computed inside each bootstrap replicate, so its SE accounts
for the covariance of the two positions.

**Per-colony uncertainty: angular block bootstrap.** Nuclei of a colony are
spatially correlated, so resampling nuclei would understate the error. The colony
is cut into `n_sectors` angular sectors (12) and the sectors are resampled with
replacement; every quantity is recomputed per replicate and its SE is the SD over
replicates. A boundary that is ragged or off-center around the colony therefore
has a large SE (on synthetic colonies a ±40 µm angular wobble raises the SE of
x50 about eightfold). A value is used only when at least `min_support` of the
replicates find it.

**Border roughness and angular registration.** A radial profile averages over
angle, so a border that is sharp locally but wanders in radius around the colony
reads as a gradient, smoother with every colony averaged. For every marker with an
accepted gate-call boundary, the colony is cut into `roughness.n_sectors` sectors
(16) and the border position is fitted in each, with the slope shared across the
colony's sectors (one free parameter per sector keeps ~150 nuclei well determined;
slope and positions are re-estimated alternately). The sector positions are
modeled as m + a cos θ + b sin θ + residual: the first harmonic (`decenter_um`) is
what an off-center pattern or colony-center fit produces and is not roughness;
`roughness_um` is the residual SD with the sectors' own fitting variance removed.
Then every nucleus is re-expressed relative to its own sector's border and the
transition is refitted: `width_registered_um` against `width_unregistered_um`
(same nuclei) answers whether the gradient is graded at each place (ratio ≈ 1) or
mostly the border's wandering (ratio ≪ 1). On synthetic colonies the method returns
roughness 21.6 µm for a border wandering with SD 21.2 µm (registration halves its
apparent width), decentering 24.7 µm and roughness 0.4 µm for a pattern 25 µm off
center, and a ratio of 0.99 for a truly graded 60 µm transition. Sixteen sectors
sample the border every ~100 µm of arc; roughness at the scale of single cells stays
inside each sector's width. The SEs of the angular statistics come from resampling
the sectors.

**Across colonies.** The per-colony SE says how well one colony determines a
number. Conditions are compared on the spread across replicates: the summary
gives mean ± SE and a t-based 95 % CI across colonies, or across wells (colonies
averaged within each well first) when the well is the replicate (§5).

**When a colony has no boundary.** The acceptance rules above mean a colony can
lack a boundary that its well-mates have. That is information, and the figures
state which case it is rather than leaving a gap unexplained:

* **Not resolved by the bootstrap.** The full data give a crossing, but fewer
  than `min_support` of the angular-block replicates do. The domain is a
  scattered set of positive cells whose apparent crossing depends on which
  sectors are sampled. Example: HOXB1 at 100 ng/mL WNT3A on
  `hd_micropatterns_bcat`, 7.6 % HOXB1+ nuclei (20 % in the outer 50 µm), the
  crossing found in 30-76 % of replicates; at 300 ng/mL 98 % of rim nuclei are
  positive and every replicate finds it.
* **One marker has no transition.** E.g. a well that is that marker's negative
  control.
* **The transitions do not face each other** in this colony.

Such colonies count in `n_colonies_rejected` and never enter a mean as zero or
as a guess.

### Does the binning shape the result?

`boundary.bin_check` (default `[fixed:5, fixed:20, quantile:20]`): every colony's positions and widths (x50, width,
crossing, overlap, peak) are refitted, point estimates only, with 5 and 20 µm
bins and with 20 equal-count (quantile) bins, and compared with the default
10 µm fit. The shift is reported in µm and in units of the default's bootstrap
SE: a shift inside ± 1-2 SE means the binning does not matter at the
measurement's own precision. Equal-count bins are narrow at the edge and wide
in the interior; on synthetic colonies with a known interior peak at 175 µm,
fixed bins of 5-20 µm recover it within ~1 µm, while 20 quantile bins pull it
~5 µm outward toward the edge. The pipeline therefore fits on fixed-width bins
(the unequal number of nuclei per bin is handled by the count-weighted fits) and
uses quantile bins only for the per-colony display.

What to expect on real data (`hd_micropatterns_bcat`): domain-boundary
positions, the facing-pair crossing and peak positions move < 0.5 SE; 95 % of
median shifts across all metrics and schemes are within 2 SE. Widths of
gate-call transitions are the most bin-sensitive (20 µm bins widened OTX2
transitions by 9-14 µm, ~2.5 SE): coarse bins smear a step. Equal-count bins
also lose weak interior peaks (found in 12 of 25 colonies where fixed bins find
them), because the few wide interior bins average over the peak.

### Nuclear density per bin

Nuclei in a distance-from-edge bin ÷ the colony
area in that bin (from the colony mask's pixel distances to the edge, or the
annulus of the fitted circle), per 1000 µm²: `density_colonies.csv`, averaged
over the replicate unit by condition in `density_profiles.csv`. The center of a
colony differs from its rim in density as well as fate, so density is the first
covariate to look at before attributing a radial difference to signaling. Each
nucleus counts once however many markers it carries. Bins with less than 5 % of
the largest bin's area (the last sliver at the center) are left empty, because a
handful of nuclei over a tiny area gives an unstable ratio. The density strip
under each colony profile draws one bar per bin with the bar width equal to the
bin width, so an unequal binning is visible at a glance.

### Saturated nuclei

A nucleus whose median pixel is at the detector ceiling in
any channel (the nuclear channel included) is clipped debris, not a bright cell;
it is dropped before scales, gates and profiles (`qc.drop_saturated_nuclei`,
`qc.saturation_ceiling` for 12-bit cameras writing 16-bit files). The ceiling
is per image and channel (`saturation_value` from extraction); for the nuclear
channel it is the lowest ceiling among that image's non-marker channels.
Left in, a clipped nucleus sets the P99 anchor of the shared scale and pulls
bin means toward the ceiling where debris collects (often the colony rim).

### Normalization captions

Every profile figure states, per marker, the value it shows ("raw nuclear
median", "÷ nuclear stain", a signaling unit), the scaling ("P1–P99 → 0–1 per
plate × panel") and that no colony is rescaled on its own. A reader can then
tell an absolute difference from a relative one without the methods section.

## 10e. Effects, trends, variance components and power

> **In plain words.** For every per-colony number: how different is each condition from the reference (with a CI), does it change steadily with dose or time (a slope with a CI), how much of the variation sits at the colony, well and experiment levels, and how many experiments a given effect would need.

The `effects` step takes every accepted per-colony number (boundary and roughness
metrics, profile features) and answers four questions the same way for each.

**How much varies at each level?** Nested method-of-moments variance components,
pooled over conditions: colonies within a well (`sd_colony`), wells within an
experiment beyond what their colonies explain (`sd_well`), experiments beyond
their wells (`sd_experiment`). A level that is not replicated in the data (one
well per condition, one differentiation) is reported as not estimable (NaN),
never as zero. Simulated designs (6 experiments × 3 wells × 5 colonies) recover
SDs of 8 / 4 / 6 as 8.0 / 4.1 / 5.3.

**How large is the effect?** Each condition against a reference (the lowest level
of a numeric condition column within its group, or `effects.reference`): the
difference of replicate means, a Welch 95 % CI, Hedges' g, and a permutation p
(exact when the labelings are few). When a comparison has fewer than two
replicates per side at the chosen unit it steps down (experiment → well →
colony) and is flagged `pseudo_replicated`: technical replicates measure the
precision of these colonies, not reproducibility.

**Does it change with dose, time or size?** For numeric condition columns
(`trend_columns: auto`), the OLS slope of replicate values against x, or against
log10 x when the positive levels span ≥ 10× (a zero level is then left out of the
fit and the table says so). The 95 % CI comes from a stratified bootstrap of
replicates within each level, p from permuting level labels across replicates.
Columns listed in `exponential_columns` also get y = c + A·exp(−x/τ) with a
bootstrap CI for τ (the decay time constant of a time course). On synthetic
images with a boundary moving 15 µm per decade of dose, three differentiations
with ±6 µm offsets and ±3 µm well offsets, the pipeline reports 14.4 µm/decade
[10.1, 18.5] with n = 9 experiment-dose replicates.

**How many experiments are needed?** With E experiments per condition, W wells
per experiment and C colonies per well the variance of a condition mean is
σ_E²/E + σ_W²/(E·W) + σ_C²/(E·W·C). The minimum detectable difference between
two conditions is (t₁₋α/₂ + t_power)·√(2·var) with df = 2(E − 1); in simulation
the stated 80 % power is achieved (76 % over 600 simulated studies). When σ_E is
not estimable it is taken as 0 and labeled a lower bound;
`power.assumed_experiment_sd` adds a scenario with an explicit assumption, and
`power.targets` reports the power for named effect sizes.

---

## 11. Assumptions and limitations

* **Exchangeability, not normality.** The permutation tests assume only that
  replicates are exchangeable under the null, which random assignment of
  conditions to wells provides.
* **Power comes from replicates.** Thousands of nuclei make each colony's curve
  precise; they cannot replace colonies and wells. Five colonies per condition
  resolves p ≈ 0.008 at best.
* **One shared scale per scope** assumes matched staining and imaging within the
  scope (§3).
* **One image, one colony.** Each imaging position must be centered on one
  pattern; a field showing two whole patterns is flagged `neighbor` when the
  second is at least half the size of the first.
* **2-D geometry.** The radial coordinate is lateral; 3-D masks are projected to
  their footprint. Profiles of 2-D and 3-D segmentations of the same colonies
  are analyzed separately and never pooled.
* **Segmentation propagates.** Everything inherits segmentation quality. The
  segmentation and colony overlays exist to be looked at.
* **Projections mix planes.** In a maximum projection, signal above and below a
  nucleus is assigned to it. This matters most for signaling readouts: it
  lowers N:C and raises the "nuclear" level of membrane-rich proteins such as
  β-catenin. Segment in 3-D when the absolute N:C matters.
* **One differentiation is one replicate of the batch.** With one
  differentiation, every interval is across colonies (or wells) of that batch;
  between-experiment variance is not estimable and power estimates are lower
  bounds. Conclusions describe these colonies until a second, independent
  differentiation reproduces them (`stats.experiment_column`).
* **Per-colony SE is not reproducibility.** The angular-block bootstrap says how
  well one colony determines a number. It says nothing about the next colony;
  the spread across colonies, wells and experiments does.
* **Gate-based widths depend on the gate and on the bins.** A gate set from a
  poor negative control shifts x50 of the gate calls; coarse bins widen a step.
  Intensity-based x50 and widths are the cross-check.
* **A failed stain is excluded at the plate map.** Leave that channel out of the
  stain set (CONFIG_GUIDE.md §9) rather than filtering afterwards, so it never
  enters a scale, a reference or a pooled table.

---

## 12. Quick reference

| Output                                                 | Question                                                                     | How to read it                                                                                           |
| ------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `figures/compare/profiles_um_*.png`                    | do conditions differ, and how far from the edge?                             | condition means ± 95 % CI across replicates; global p in the title; gray bar = where they differ         |
| `figures/compare/profiles_norm_*.png`                  | the same, relative to colony size                                            | x = fraction of the radius from the edge                                                                 |
| `figures/compare/pairwise_*.png`                       | which pairs differ?                                                          | BH q per marker × pair; * = q < 0.05                                                                     |
| `figures/compare/features*.png`                        | which interpretable property differs?                                        | one point per replicate, mean ± CI; Kruskal-Wallis q in the title                                        |
| `figures/compare/similarity*.png`                      | are colonies of a condition alike?                                           | diagonal blocks = coherent replicates                                                                    |
| `figures/profiles/colonies_*.png`                      | is every colony like its well-mates?                                         | thin lines = colonies, bold = well mean, dotted × = flagged colony                                       |
| `figures/colony_profiles/*.png`                        | what does one colony look like?                                              | that colony's own bins in µm; band = CI over its nuclei (descriptive)                                    |
| `figures/qc/colony_overlays_*.png`                     | was every colony found correctly?                                            | orange = edge, dashed = circle, × = outside colony, flags in the corner                                  |
| `tables/compare_global.csv`                            | the headline numbers                                                         | one row per marker × grid × readout                                                                      |
| `figures/signaling_summary/*.png`                      | how strong is a pathway per condition?                                       | one point per colony (whole, edge band, center); dashed line = reference                                 |
| `figures/signaling/ring_qc_*.png`                      | was the N:C cytoplasm measured in the right place?                           | orange = nucleus core, sky blue = ring inside cell bodies                                                |
| `figures/boundary/pair_*.png`                          | where do two facing domains meet, and do they overlap?                       | one point per colony ± its bootstrap SE; crossing, overlap (< 0 = gap), double positives vs independence |
| `figures/boundary/peak_*.png`                          | where is the signaling maximum relative to the boundaries?                   | peak position, peak / trough, offsets; marker shape = stain panel                                        |
| `figures/boundary/roughness_*.png`, `registered_*.png` | is a wide transition real or a wandering border?                             | roughness in µm; registered vs unregistered width (ratio ≈ 1 = truly graded)                             |
| `figures/boundary/bin_check.png`                       | does the binning move the numbers?                                           | shift per scheme; grey band = ± median SE                                                                |
| `figures/effects/*.png`                                | how big is the effect, does it follow dose, how many experiments are needed? | difference vs reference with CI; slope per decade; MDD vs number of experiments                          |
| `figures/profiles/density_um.png`                      | is density different where fates differ?                                     | nuclei per 1000 µm² vs distance, by condition                                                            |
| `tables/boundary_summary.csv`, `effects_*.csv`         | the numbers behind the claims                                                | `primary = True` rows are at the chosen replicate unit                                                   |

---

## 13. References

- Warmflash A, Sorre B, Etoc F, Siggia ED, Brivanlou AH. A method to recapitulate
  early embryonic spatial patterning in human embryonic stem cells. *Nat Methods*
  2014;11:847-854.
- Etoc F, Metzger J, Ruzo A, Kirst C, Yoney A, Ozair MZ, Brivanlou AH, Siggia ED.
  A balance between secreted inhibitors and edge sensing controls gastruloid
  self-organization. *Dev Cell* 2016;39:302-315.
- Ramsay JO, Silverman BW. *Functional Data Analysis*, 2nd ed. Springer, 2005.
- Good P. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed.
  Springer, 2005.
- Westfall PH, Young SS. *Resampling-Based Multiple Testing.* Wiley, 1993.
- Nichols TE, Holmes AP. Nonparametric permutation tests for functional
  neuroimaging: a primer with examples. *Hum Brain Mapp* 2002;15:1-25.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *J R Stat Soc B* 1995;57:289-300.
- Bonett DG, Wright TA. Sample size requirements for estimating Pearson,
  Kendall and Spearman correlations. *Psychometrika* 2000;65:23-28.
- Kåsa I. A circle fitting procedure and its error analysis. *IEEE Trans Instrum
  Meas* 1976;25:8-14.
- Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization
  for cellular segmentation. *bioRxiv* 2025.
