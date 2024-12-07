#!/usr/bin/env python3
"""
Deployment Script
Deploy fine-tuned model to production
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deploy_local_vllm(model_path: str, port: int = 8000, tensor_parallel: int = 1):
    """Deploy model using vLLM locally"""
    logger.info("Deploying model with vLLM...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Port: {port}")
    logger.info(f"Tensor Parallel: {tensor_parallel}")

    src_dir = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_dir))

    try:
        from src.serving.vllm_server import vLLMServer

        server = vLLMServer(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel
        )

        logger.info(f"Starting vLLM server on port {port}...")
        server.run(port=port)

    except ImportError as e:
        logger.error("vLLM not installed or import failed")
        logger.error("Please run: pip install vllm")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)


def deploy_docker(model_path: str, port: int = 8000):
    """Deploy model using Docker"""
    logger.info("Deploying model with Docker...")

    docker_file = Path(__file__).parent.parent / "docker" / "Dockerfile.serving"

    if not docker_file.exists():
        logger.error(f"Docker file not found: {docker_file}")
        sys.exit(1)

    # Build Docker image
    logger.info("Building Docker image...")
    build_cmd = [
        "docker", "build",
        "-f", str(docker_file),
        "-t", "llm-finetuner-serving",
        str(Path(__file__).parent.parent)
    ]

    try:
        subprocess.run(build_cmd, check=True)
        logger.info("Docker image built successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e}")
        sys.exit(1)

    # Run Docker container
    logger.info(f"Starting Docker container on port {port}...")
    run_cmd = [
        "docker", "run",
        "--gpus", "all",
        "-p", f"{port}:8000",
        "-v", f"{model_path}:/app/models",
        "llm-finetuner-serving"
    ]

    try:
        subprocess.run(run_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker run failed: {e}")
        sys.exit(1)


def deploy_cloud(
    model_path: str,
    cloud_provider: str = "aws",
    region: str = "us-east-1"
):
    """Deploy model to cloud (AWS/GCP)"""
    logger.info(f"Deploying to {cloud_provider.upper()}...")

    if cloud_provider == "aws":
        deploy_aws(model_path, region)
    elif cloud_provider == "gcp":
        deploy_gcp(model_path, region)
    else:
        logger.error(f"Unsupported cloud provider: {cloud_provider}")
        sys.exit(1)


def deploy_aws(model_path: str, region: str):
    """Deploy to AWS SageMaker or EC2"""
    logger.info("AWS deployment not yet implemented")
    logger.info("Please deploy manually using:")
    logger.info("1. Upload model to S3")
    logger.info("2. Create EC2 instance with GPU")
    logger.info("3. Install dependencies and run vLLM server")
    logger.info("\nOr use AWS SageMaker for managed deployment")


def deploy_gcp(model_path: str, region: str):
    """Deploy to GCP Vertex AI or Compute Engine"""
    logger.info("GCP deployment not yet implemented")
    logger.info("Please deploy manually using:")
    logger.info("1. Upload model to GCS")
    logger.info("2. Create Compute Engine instance with GPU")
    logger.info("3. Install dependencies and run vLLM server")
    logger.info("\nOr use Vertex AI for managed deployment")


def test_deployment(host: str = "localhost", port: int = 8000):
    """Test the deployed model"""
    import requests

    url = f"http://{host}:{port}/health"
    logger.info(f"Testing deployment at {url}...")

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logger.info("Deployment successful! Health check passed.")
            return True
        else:
            logger.error(f"Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to server: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy fine-tuned model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--method", type=str, default="local",
                       choices=["local", "docker", "aws", "gcp"],
                       help="Deployment method")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for serving")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--region", type=str, default="us-east-1",
                       help="Cloud region (for cloud deployments)")
    parser.add_argument("--test", action="store_true",
                       help="Test deployment after starting")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("LLM Model Deployment")
    logger.info("="*60)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Deploy based on method
    if args.method == "local":
        deploy_local_vllm(
            str(model_path),
            port=args.port,
            tensor_parallel=args.tensor_parallel
        )
    elif args.method == "docker":
        deploy_docker(str(model_path), port=args.port)
    elif args.method in ["aws", "gcp"]:
        deploy_cloud(str(model_path), cloud_provider=args.method, region=args.region)

    # Test deployment if requested
    if args.test:
        import time
        logger.info("Waiting for server to start...")
        time.sleep(5)
        test_deployment(port=args.port)


if __name__ == "__main__":
    main()
