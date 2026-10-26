# AI usage

## Tools I used, and for what

I used Codex to help turn the written specification into a first implementation of the sklearn
pipeline, the history calculation, and the evaluation functions. I used the public pytest suite and
an extra seed-0 run to check the implementation and record the numbers in RESULTS.md.

## AI-suggested code I rejected or fixed in add_history_features or time_split

An initial approach accumulated history row by row after sorting. I changed it so that every parcel
at the same timestamp receives the history from before that timestamp; otherwise same-minute parcels
could incorrectly count one another as prior. The public history test checked the basic rate logic,
and I checked the equal-timestamp rule against the task's `history_ties_are_not_prior` requirement.

## How I checked that my model has no leak

I excluded parcel_id, courier_notes, delivered_at, and failed from FEATURE_COLUMNS, and the model's
time-split ROC AUC (0.654) is far below the task's stated leak warning level of 0.9.
