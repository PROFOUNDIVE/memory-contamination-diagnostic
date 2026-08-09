# Rootless-local post-BCT review contract

All inputs, outputs, sentinels, QA envelopes, publication receipts, and final indexes governed by
this contract are `local_rootless_non_authoritative`. Rootless receipts never enter authoritative
or scientific evidence indexes.

After screening, Freeze B may select exactly two common-strict probes per task. The bounded BCT may
then evaluate only the four fixed candidate classes through the four fixed baselines. Successful
local completion seals `BCT_COMPLETED_REVIEW_REQUIRED` internally and exposes only
`LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED`. It does not select a policy, establish readiness, or permit
Pilot-B.

Review must reconcile the signed receipt manifests, ledger head, stage and final terminals, state
inventory, redacted publication, T7 locator, and QA envelope without reopening `.env`, replaying a
stage, or making a provider call. Missing, reordered, additional, cross-profile, or hash-drifted
evidence fails closed. Rootless and historical/v1 artifacts remain schema-disjoint.

Same-UID cooperation and Docker `USER` are not host isolation. Provider billing is externally
unbounded by the cooperative local ledger. The theoretical Google Drive sources are observed only
through the point-in-time namespace-local read-only predicate; permission repair, ownership
changes, remounting, copying, mirroring, and local substitution are forbidden.
