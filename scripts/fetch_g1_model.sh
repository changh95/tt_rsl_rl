#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Fetch the Unitree G1 MuJoCo model (MuJoCo Menagerie, BSD-3-Clause, (c) Unitree Robotics) pinned to one commit, into .run/assets/ (git-ignored; 38 MB incl. meshes).
# Only the unitree_g1 directory is checked out (sparse clone). The files are DATA: they are read by MuJoCo, nothing in them is executed.
#
#     scripts/fetch_g1_model.sh
set -eu
cd "$(dirname "$0")/.."
COMMIT=370c43965323c0944b433b202aac5058b5f907a9
DIR=.run/assets/mujoco_menagerie
mkdir -p .run/assets
if [ ! -d "$DIR/.git" ]; then
  git clone --quiet --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DIR"
fi
git -C "$DIR" sparse-checkout set unitree_g1
git -C "$DIR" fetch --quiet --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$DIR" checkout --quiet "$COMMIT"
echo "G1 model at $DIR/unitree_g1 (menagerie commit $(git -C "$DIR" rev-parse --short HEAD))"
