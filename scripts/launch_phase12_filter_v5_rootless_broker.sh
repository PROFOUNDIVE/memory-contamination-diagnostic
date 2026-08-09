#!/bin/sh
set -eu
umask 077

fail() {
    printf '%s\n' ROOTLESS_PRELAUNCH_INVALID >&2
    exit 64
}

close_inherited_fds() {
    [ -d "/proc/$$/fd" ] || fail
    for path in /proc/$$/fd/*; do
        fd=${path##*/}
        case "$fd" in
            0|1|2) ;;
            *[!0-9]*|'') fail ;;
            *) eval "exec ${fd}>&-" 2>/dev/null || fail ;;
        esac
    done
}

for name in OPENAI_API_KEY HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
    PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS \
    SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
    eval "present=\${${name}+yes}"
    [ "${present:-}" != yes ] || fail
done

python=
repo_root=
state_home=
authority=
attempt_id=
action=
acknowledged=no
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python|--repo-root|--state-home|--authority|--attempt-id)
            [ "$#" -ge 2 ] || fail
            name=${1#--}
            name=$(printf '%s' "$name" | tr - _)
            eval "$name=\$2"
            shift 2
            ;;
        run-screening|run-bct)
            [ -z "$action" ] || fail
            action=$1
            shift
            ;;
        --acknowledge-local-non-authoritative)
            acknowledged=yes
            shift
            ;;
        *) fail ;;
    esac
done

[ -n "$python" ] && [ -n "$repo_root" ] && [ -n "$state_home" ] || fail
[ -n "$authority" ] && [ -n "$attempt_id" ] && [ "$acknowledged" = yes ] || fail
case "$python:$repo_root:$state_home:$authority" in /*:/*:/*:/*) ;; *) fail ;; esac
[ -x "$python" ] && [ ! -L "$python" ] && [ -d "$repo_root" ] && [ ! -L "$repo_root" ] || fail
[ -d "$state_home" ] && [ ! -L "$state_home" ] && [ -f "$authority" ] && [ ! -L "$authority" ] || fail
case "$action" in run-screening) stage=screening ;; run-bct) stage=bct ;; *) fail ;; esac

close_inherited_fds
exec /usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin "$python" -B -I -m memcontam.cli \
    phase12 filter-v5-rootless --repo-root "$repo_root" --state-home "$state_home" \
    broker-runtime --attempt-id "$attempt_id" --stage "$stage" --authority "$authority" \
    --worker-fd 3
