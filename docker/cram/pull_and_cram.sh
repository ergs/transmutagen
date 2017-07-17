#!/bin/bash

# This is a separate script so that it can pull any updates from git correctly
# without rebuilding the docker container. All arguments are passed through to
# python -m transmutagen.cram.

set -e
set -x

git pull

python -m transmutagen.cram "$@"
