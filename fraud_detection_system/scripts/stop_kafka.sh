#!/bin/bash

# Stop Kafka cluster

echo "Stopping Kafka cluster..."

cd "$(dirname "$0")/.."

docker-compose -f docker/docker-compose.kafka.yaml down

echo "Kafka cluster stopped."