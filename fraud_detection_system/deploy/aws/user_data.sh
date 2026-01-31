#Best for PoC or small-scale production. --> EC2 + Docker Compose
#!/bin/bash

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Login to ECR (if using private registry)
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

# Clone Repo (or pull images directly)
# git clone https://github.com/your-repo/fraud-detection-system.git
# cd fraud_detection_system

# Pull latest images
docker pull <your-dockerhub-user>/fraud-detection-api:latest
docker pull <your-dockerhub-user>/fraud-detection-streaming:latest

# Start services
docker-compose up -d