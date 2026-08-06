#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./run_rps_homography_grasp_d455.sh --gesture-serial "" "$@"
