#!/bin/bash

# Start Kafka using Docker Compose

echo "Starting Kafka cluster..."

cd "$(dirname "$0")/.."

docker-compose -f docker/docker-compose.kafka.yaml up -d

echo "Waiting for Kafka to be ready..."
sleep 10

# Check if Kafka is ready
docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Kafka is ready!"
    echo ""
    echo "Services:"
    echo "  - Kafka: localhost:9092"
    echo "  - Zookeeper: localhost:2181"
    echo "  - Kafka UI: http://localhost:8080"
    echo "  - Schema Registry: http://localhost:8081"
else
    echo "Kafka is still starting. Please wait..."
fi