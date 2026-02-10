# Ranking evaluation note (NDCG@K)

The Kaggle Criteo train set does not provide explicit "query/session" IDs.
To approximate a ranking scenario offline, we build **pseudo groups**:

- `group_id = C1 + "_" + time_bucket`
- `time_bucket` is derived from the row order (log order) into buckets of size 50

This produces small groups where the model ranks candidates within a bucket.
You can adjust the bucket size (e.g., 20/50/100) in `src/cli.py` to trade off:
- smaller buckets → closer to session-like ranking but noisier
- larger buckets → more stable but less session-like

If you have a dataset with true request/session IDs, replace `group_id` with your real grouping.