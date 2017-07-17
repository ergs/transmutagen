#!/bin/bash
set -e
set -x

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

docker build -f Dockerfile . -t cram
docker run -v "$parent_path/../../logs":/logs -v "$parent_path/../../CRAM_cache":/home/.transmutagen/CRAM_cache cram --save-cache "$@"
