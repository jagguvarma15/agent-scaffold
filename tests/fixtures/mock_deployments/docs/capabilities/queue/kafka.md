---
id: queue.kafka
kind: queue
provides: [event_stream]
env_vars: [KAFKA_BOOTSTRAP_SERVERS]
docker:
  service: kafka
  image: bitnami/kafka:3.7
  ports: ["9092:9092"]
probe: kafka_metadata
docs: |
  Kafka event streaming.
---

# Capability: queue.kafka

Test fixture body.
