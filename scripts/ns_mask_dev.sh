#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# ns_mask_dev.sh -- DECISIONS ADR-0005 (option II): give a job a /dev/tenstorrent that contains ONLY the chips it may use.
#
#   unshare --user --map-root-user --mount -- scripts/ns_mask_dev.sh \
#       --dev-dir /dev/tenstorrent --keep "16 17" --stage .run/ns_stage [--require-chardev] -- CMD [ARGS...]
#
# It must run INSIDE a fresh user+mount namespace (tt_run.sh arranges that with `unshare`); the mounts below exist only in that
# namespace, so no other process on the host (other people's jobs on the other trays) sees or is affected by them. No sudo.
#
# What it does:
#   1. mount a private tmpfs on STAGE and bind-mount each kept node (DEV_DIR/N) onto a placeholder file STAGE/N;
#   2. `mount --move` STAGE on top of DEV_DIR, which hides EVERY other entry of DEV_DIR (other trays' nodes);
#   3. verify the resulting view lists exactly the kept names (and, with --require-chardev, that each is a character device);
#   4. exec CMD.
# Effect: UMD's enumeration (a directory listing of /dev/tenstorrent) only ever sees the kept nodes, so it never open()s another
# tray's chip. The kmd resource locks are kept by the kernel driver, so they still arbitrate across namespaces (UMD kmd_mutex.hpp).
set -euo pipefail

DEV_DIR=""
KEEP=""
STAGE=""
REQUIRE_CHARDEV=0

die() { echo "ns_mask_dev: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev-dir)         DEV_DIR="$2"; shift 2 ;;
        --keep)            KEEP="${2//,/ }"; shift 2 ;;
        --stage)           STAGE="$2"; shift 2 ;;
        --require-chardev) REQUIRE_CHARDEV=1; shift ;;
        --)                shift; break ;;
        *)                 die "unknown argument '$1'" ;;
    esac
done
[[ -n "$DEV_DIR" && -n "$KEEP" && -n "$STAGE" ]] || die "--dev-dir, --keep and --stage are required"
[[ $# -gt 0 ]] || die "no command given"
[[ -d "$DEV_DIR" ]] || die "dev dir '$DEV_DIR' does not exist"
for n in $KEEP; do
    [[ "$n" =~ ^[0-9]+$ ]] || die "kept node '$n' is not a number"
    [[ -e "$DEV_DIR/$n" ]] || die "kept node '$DEV_DIR/$n' does not exist"
    if [[ "$REQUIRE_CHARDEV" -eq 1 && ! -c "$DEV_DIR/$n" ]]; then die "kept node '$DEV_DIR/$n' is not a character device"; fi
done
# Refuse to run outside a user namespace (an identity uid_map means the initial one): without it these mounts would be real
# host-wide mounts if this were ever run as root. tt_run.sh always enters a fresh user+mount namespace first.
if grep -qE '^[[:space:]]*0[[:space:]]+0[[:space:]]+4294967295' /proc/self/uid_map; then
    die "not inside a user namespace (run through tt_run.sh)"
fi

mount --make-rprivate /   # belt and braces: nothing we mount may propagate out of this namespace
mkdir -p "$STAGE"
mount -t tmpfs -o mode=0755,size=64k tmpfs "$STAGE"
for n in $KEEP; do
    : > "$STAGE/$n"
    mount --bind "$DEV_DIR/$n" "$STAGE/$n"
done
mount --move "$STAGE" "$DEV_DIR"

# verify the view the job will have
seen="$(ls -A "$DEV_DIR" | sort -n | tr '\n' ' ')"
want="$(printf '%s\n' $KEEP | sort -n | uniq | tr '\n' ' ')"
[[ "$seen" == "$want" ]] || die "masked view of '$DEV_DIR' is '$seen' but expected '$want'"
if [[ "$REQUIRE_CHARDEV" -eq 1 ]]; then
    for n in $KEEP; do [[ -c "$DEV_DIR/$n" ]] || die "'$DEV_DIR/$n' is not a character device after masking"; done
fi
echo "ns_mask_dev: $DEV_DIR now shows only: $want" >&2
exec "$@"
