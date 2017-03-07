#!/bin/bash
set -e
set -x

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

docker build -f Dockerfile-base . -t origen-base
docker build -f Dockerfile --no-cache . -t origen
docker run -v "$parent_path/../data":/data -v "$parent_path/../logs":/logs origen "$@"
