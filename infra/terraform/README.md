# Infrastructure as Code (`infra/terraform`)

Terraform modules for provisioning the Single-Pass 3D Reconstruction Platform on AWS:
- `vpc/`: Multi-AZ networking, public/private subnets, NAT gateways.
- `rds/`: Amazon RDS PostgreSQL 18 with PostGIS extension.
- `elasticache/`: Redis cluster.
- `s3/`: Bucket configurations for raw flight videos, interim reconstruction scratch, and 3D models.
- `sqs/`: Job orchestration queues and Dead-Letter Queues (DLQ).
- `eks/`: Kubernetes cluster, Karpenter dynamic GPU node pools, and NVIDIA GPU Operator.
- `cloudfront/`: CDN distribution for fast 3D Tiles streaming.
