#!/bin/bash
docker build -f Dockerfile-base . -t origen-base
docker build -f Dockerfile --no-cache . -t origen
docker run origen
