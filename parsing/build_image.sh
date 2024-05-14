#/bin/bash

# this script assumes running on M1 Mac
export DOCKER_DEFAULT_PLATFORM=linux/amd64

DOCKER_IMAGE=mendeza/pdk:parsing-v0.0.1

echo "Running docker buildx build --platform linux/amd64 . -t $DOCKER_IMAGE --no-cache"

docker buildx build --platform linux/amd64 . -t $DOCKER_IMAGE --no-cache 

echo "Running  docker push $DOCKER_IMAGE..."

docker push $DOCKER_IMAGE 