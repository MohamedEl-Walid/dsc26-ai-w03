# Results: Tawseel failed-delivery baseline

## Baseline table

Last 20% by `created_at`, using `make_parcels(4000, seed=0)`. The model is a logistic-regression
pipeline with decision-time parcel details and point-in-time customer history. I did not include a
hand-written rule because it would not be a fair replacement for the requested prior-only baseline.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---|---|---|---|
| Dummy (prior) | 0.500 | 0.220 | 0.172 | 0.220 |
| My model | 0.654 | 0.385 | 0.162 | 0.220 |

## Three sentences for Sherif

This baseline ranks failed deliveries better than the prior-only reference, improving PR-AUC from
0.220 to 0.385 on the most recent 20% of the synthetic parcels.  The improvement is useful but
modest, so it is suitable for helping analysts prioritize a queue rather than making automatic
delivery decisions.  The evaluation is an honest time split and excludes post-attempt fields, but
we still need real production data and monitoring before relying on it operationally.
