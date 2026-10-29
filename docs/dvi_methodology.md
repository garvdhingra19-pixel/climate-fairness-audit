# Data Visibility Index (DVI): methodology

**Question:** how well is each district "seen" by climate data? The DVI is a *measurement-coverage*
score, not a quality ranking of districts and not a judgement of any agency.

## Components

| Component | Raw column | Direction | Log | Default weight |
|---|---|---|---|---|
| station_density_area | stations per 1,000 km2 | higher = more visible | yes | 0.20 |
| station_density_pop | stations per 100,000 people | higher | yes | 0.20 |
| distance_to_station | km from district centroid to nearest station | lower | yes | 0.20 |
| survey_sample | survey sample size | higher | yes | 0.15 |
| completeness | share of valid readings | higher | no | 0.15 |
| satellite_quality | cloud-free fraction | higher | no | 0.10 |

## Computation
1. Optional `log1p` transform for heavy-tailed counts and distances.
2. Winsorise at the 5th and 95th percentiles, then min-max scale to [0, 1].
3. Flip components where lower is better.
4. Weighted mean (weights renormalised to sum to 1), scaled to 0-100.

## Missing data policy
A missing component is **skipped** and the remaining weights are renormalised for that district.
`n_components_used` records how many components contributed, so thin evidence is visible.
Districts with no station are not "missing" for density (they score 0) but have no completeness.

## Weight sensitivity
`sensitivity_analysis` draws weights from a Dirichlet distribution centred on the defaults and
reports Spearman rank correlation and quintile stability. Conclusions should be reported only if
they hold across these perturbations.

## Known limitations
- Centroid-to-station distance ignores terrain, wind and within-district population spread.
- Default weights are a judgement call; document and justify any change.
- District boundaries change over time; always use a verified crosswalk when joining real data.
