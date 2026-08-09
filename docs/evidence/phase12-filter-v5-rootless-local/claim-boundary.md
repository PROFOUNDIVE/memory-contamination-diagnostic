# Rootless-local claim boundary

Every artifact in this directory and every associated receipt has profile
`local_rootless_non_authoritative`. These artifacts are local operational records only. They are
excluded from authoritative evidence indexes, scientific admission, claim aggregation, Filter-v5
selected-policy evidence, Pilot-B authorization, and all historical build/v1 evidence.

The historical and canonical evidence remains provider-free. An optional, separately redacted
rootless-local receipt may report paid calls, but that does not turn the receipt into benchmark,
scientific, causal, production, manuscript, or authoritative evidence.

The workflow provides cooperative same-UID accounting and logical/session separation. It does not
provide host root isolation. Running under the same UID, or setting Docker `USER`, must not be
described as host isolation, independent authority, tamper resistance, or a security boundary.

The three theoretical authority files may retain mode `0664` below `0775` ancestors only when the
exact descriptor-bound bytes are observed through both the selected mount's `ro` option and
`ST_RDONLY` in the current namespace. That observation is point-in-time, namespace-local,
same-UID-bypassable, and non-authoritative. Do not `chmod`, `chown`, remount, copy, mirror, repair
permissions on, or substitute a local copy for any Google Drive authority file.

No rootless receipt authorizes Pilot-B, readiness, selected-policy conversion, or further provider
egress. A valid local BCT stops at `LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED`.
