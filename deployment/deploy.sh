#!/bin/bash

# E-commerce Analytics Platform Deployment Script

set -e

echo "🚀 Starting deployment of E-commerce Analytics Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
VERSION=${VERSION:-latest}

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_status "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_status "Docker Compose is installed"
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from .env.example..."
        cp .env.example .env
        print_warning "Please update .env file with your configuration"
        exit 1
    fi
    print_status ".env file exists"
}

# Build Docker images
build_images() {
    echo "Building Docker images..."
    
    # Build main application
    print_status "Building main application image..."
    docker build -t ecommerce-analytics:${VERSION} -f Dockerfile .
    
    # Build API service
    print_status "Building API service image..."
    docker build -t ecommerce-analytics-api:${VERSION} -f Dockerfile.api .
    
    # Build scheduler service
    print_status "Building scheduler service image..."
    docker build -t ecommerce-analytics-scheduler:${VERSION} -f Dockerfile.scheduler .
    
    print_status "All images built successfully"
}

# Tag and push images (if registry is specified)
push_images() {
    if [ -n "$DOCKER_REGISTRY" ]; then
        echo "Pushing images to registry: $DOCKER_REGISTRY"
        
        # Tag images
        docker tag ecommerce-analytics:${VERSION} ${DOCKER_REGISTRY}/ecommerce-analytics:${VERSION}
        docker tag ecommerce-analytics-api:${VERSION} ${DOCKER_REGISTRY}/ecommerce-analytics-api:${VERSION}
        docker tag ecommerce-analytics-scheduler:${VERSION} ${DOCKER_REGISTRY}/ecommerce-analytics-scheduler:${VERSION}
        
        # Push images
        docker push ${DOCKER_REGISTRY}/ecommerce-analytics:${VERSION}
        docker push ${DOCKER_REGISTRY}/ecommerce-analytics-api:${VERSION}
        docker push ${DOCKER_REGISTRY}/ecommerce-analytics-scheduler:${VERSION}
        
        print_status "Images pushed to registry"
    fi
}

# Deploy with Docker Compose
deploy_docker_compose() {
    echo "Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        print_status "Streamlit app is healthy"
    else
        print_error "Streamlit app health check failed"
    fi
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "API service is healthy"
    else
        print_error "API service health check failed"
    fi
    
    print_status "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo "Deploying to Kubernetes..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/deployment.yaml
    
    # Wait for deployments
    kubectl -n ecommerce-analytics wait --for=condition=available --timeout=300s deployment/ecommerce-app
    kubectl -n ecommerce-analytics wait --for=condition=available --timeout=300s deployment/ecommerce-api
    
    print_status "Kubernetes deployment completed"
    
    # Get service endpoints
    echo "Service endpoints:"
    kubectl -n ecommerce-analytics get svc
}

# Run post-deployment checks
post_deployment_checks() {
    echo "Running post-deployment checks..."
    
    # Check if models exist
    if [ -d "artifacts/models" ] && [ "$(ls -A artifacts/models)" ]; then
        print_status "Model artifacts found"
    else
        print_warning "No model artifacts found. Run training pipeline first."
    fi
    
    # Check data directory
    if [ -d "data" ] && [ "$(ls -A data/*.csv 2>/dev/null)" ]; then
        print_status "Data files found"
    else
        print_warning "No data files found. Please add dataset to data/ directory"
    fi
}

# Main deployment flow
main() {
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Build images
    build_images
    
    # Push images if registry is specified
    push_images
    
    # Deploy based on environment
    case $ENVIRONMENT in
        "development"|"staging")
            deploy_docker_compose
            ;;
        "production")
            if [ "$2" == "kubernetes" ]; then
                deploy_kubernetes
            else
                deploy_docker_compose
            fi
            ;;
        *)
            print_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Run post-deployment checks
    post_deployment_checks
    
    echo ""
    print_status "Deployment completed successfully! 🎉"
    echo ""
    echo "Access the application at:"
    echo "  - Streamlit UI: http://localhost:8501"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - MLflow: http://localhost:5000"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f"
}

# Run main function
main "$@"