# AWS Deployment Guide for vLLM

## Overview
This guide shows how to deploy your fine-tuned LLM using vLLM on AWS.

## Prerequisites
- AWS Account
- AWS CLI installed and configured
- Trained model ready to deploy

## Option 1: EC2 Instance (Recommended for Beginners)

### Step 1: Launch EC2 Instance

```bash
# Launch GPU instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type g5.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxx \
    --subnet-id subnet-xxxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'
```

**Recommended Instance Types:**

| Instance | GPU | VRAM | Cost/hour | Best For |
|----------|-----|------|-----------|----------|
| g5.xlarge | 1x A10G | 24GB | $1.01 | 7B models |
| g5.2xlarge | 1x A10G | 24GB | $1.21 | 7B models (more CPU) |
| g5.4xlarge | 1x A10G | 24GB | $1.62 | High traffic |
| p3.2xlarge | 1x V100 | 16GB | $3.06 | Legacy option |

### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# Install CUDA (after reboot)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Install Python and dependencies
sudo apt install -y python3.10 python3-pip
pip install vllm torch transformers
```

### Step 3: Upload Model to EC2

```bash
# Option A: Upload from local machine
scp -i your-key.pem -r ./models/llama-sql-v1 \
    ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com:~/models/

# Option B: Download from S3
aws s3 sync s3://your-bucket/models/llama-sql-v1 ~/models/llama-sql-v1

# Option C: Download from HuggingFace
git lfs install
git clone https://huggingface.co/your-username/llama-sql-v1
```

### Step 4: Start vLLM Server

```bash
# Start vLLM server
python -m vllm.entrypoints.api_server \
    --model ~/models/llama-sql-v1 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# Or use our custom server
cd llm-finetuner
python -m src.serving.vllm_server \
    --model ~/models/llama-sql-v1 \
    --port 8000
```

### Step 5: Configure Security Group

```bash
# Allow inbound traffic on port 8000
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxx \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0
```

### Step 6: Test Deployment

```bash
# Test from your local machine
curl -X POST http://ec2-xx-xx-xx-xx.compute.amazonaws.com:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Generate SQL: Find all users from 2024",
        "max_tokens": 128,
        "temperature": 0.1
    }'
```

---

## Option 2: Docker + ECR + ECS (Production)

### Step 1: Build and Push Docker Image

```bash
# Build image
docker build -f docker/Dockerfile.serving -t llm-finetuner-serving .

# Tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag llm-finetuner-serving:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-finetuner:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-finetuner:latest
```

### Step 2: Create ECS Task Definition

```json
{
  "family": "llm-finetuner-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "30720",
  "containerDefinitions": [
    {
      "name": "vllm-container",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-finetuner:latest",
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

### Step 3: Create ECS Service

```bash
aws ecs create-service \
    --cluster llm-cluster \
    --service-name llm-service \
    --task-definition llm-finetuner-task \
    --desired-count 1 \
    --launch-type EC2
```

---

## Option 3: AWS SageMaker (Fully Managed)

### Step 1: Package Model for SageMaker

```python
# package_model.py
import tarfile
import os

model_dir = "./models/llama-sql-v1"
output_file = "model.tar.gz"

with tarfile.open(output_file, "w:gz") as tar:
    tar.add(model_dir, arcname=".")

# Upload to S3
import boto3
s3 = boto3.client('s3')
s3.upload_file(output_file, 'your-bucket', 'models/model.tar.gz')
```

### Step 2: Deploy to SageMaker

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

role = sagemaker.get_execution_role()

# Create model
huggingface_model = HuggingFaceModel(
    model_data="s3://your-bucket/models/model.tar.gz",
    role=role,
    transformers_version="4.36.0",
    pytorch_version="2.1.0",
    py_version="py310",
    env={
        'HF_TASK': 'text-generation',
        'MAX_LENGTH': '512'
    }
)

# Deploy
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="llm-finetuner-endpoint"
)

# Test
result = predictor.predict({
    "inputs": "Generate SQL: Find all users"
})
print(result)
```

---

## Cost Optimization Tips

### 1. Use Spot Instances (70% savings!)

```bash
# Launch spot instance
aws ec2 request-spot-instances \
    --instance-type g5.xlarge \
    --spot-price "0.50" \
    --launch-specification file://spec.json
```

### 2. Auto-Scaling

```bash
# Scale based on CPU/GPU utilization
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/llm-cluster/llm-service \
    --min-capacity 1 \
    --max-capacity 10
```

### 3. Use S3 for Model Storage

```bash
# Store model in S3, download on startup
# Saves EBS costs
```

### 4. Reserved Instances (up to 72% savings for 1-3 year commitments)

---

## Monitoring and Logging

### CloudWatch Metrics

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Log custom metrics
cloudwatch.put_metric_data(
    Namespace='LLM-Finetuner',
    MetricData=[
        {
            'MetricName': 'TokensPerSecond',
            'Value': 55.0,
            'Unit': 'Count'
        },
        {
            'MetricName': 'GPUUtilization',
            'Value': 85.0,
            'Unit': 'Percent'
        }
    ]
)
```

---

## Production Checklist

- [ ] Set up CloudWatch alarms for GPU/CPU/Memory
- [ ] Configure auto-scaling based on load
- [ ] Set up Application Load Balancer for multiple instances
- [ ] Enable HTTPS with ACM certificate
- [ ] Set up CloudWatch Logs for debugging
- [ ] Configure IAM roles with least privilege
- [ ] Set up VPC with private subnets for security
- [ ] Enable CloudTrail for audit logging
- [ ] Set up backup strategy for models
- [ ] Configure rate limiting and authentication

---

## Cost Estimation

### Scenario: Moderate Traffic (1M requests/month)

```
EC2 g5.xlarge:
- Hourly cost: $1.01
- Monthly (24/7): $730
- Per 1M requests: $730

vLLM throughput: ~80 req/sec = 288K req/hour
1M requests = ~3.5 hours of processing
Actual cost (with auto-scaling): ~$3.50

Compare to GPT-4 API:
- $10 per 1M tokens
- Avg 500 tokens per request
- Cost: $5,000 for 1M requests

Savings: 99.3%
```

---

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check GPU
sudo apt install nvidia-utils-530
```

### Out of Memory
```bash
# Reduce max_model_len
python -m vllm.entrypoints.api_server \
    --model model \
    --max-model-len 2048
```

### Slow Inference
```bash
# Enable tensor parallelism (multi-GPU)
--tensor-parallel-size 2
```

---

## Next Steps

1. Set up CI/CD pipeline for automated deployments
2. Implement A/B testing for model versions
3. Add monitoring dashboard with Grafana
4. Configure backup and disaster recovery
5. Optimize model with quantization (GPTQ/AWQ)

---

## Resources

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/g5/)
- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
