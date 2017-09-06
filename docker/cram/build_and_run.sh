#!/bin/bash
set -e
set -x

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

docker build -f Dockerfile . -t ergs/cram
docker run -v "$parent_path/../../logs":/root/transmutagen/logs -v "$parent_path/../../plots":/root/transmutagen/plots -v "$parent_path/../../CRAM_cache":/root/.transmutagen/CRAM_cache cram --save-cache "$@"
