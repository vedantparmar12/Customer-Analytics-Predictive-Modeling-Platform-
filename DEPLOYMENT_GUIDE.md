# E-commerce Analytics Platform - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the E-commerce Analytics Platform to production using Docker and Kubernetes.

## Architecture

The platform consists of the following services:

1. **Main Application** - Streamlit dashboard for analytics and visualization
2. **API Service** - FastAPI for real-time predictions
3. **MLflow** - Model tracking and versioning
4. **Redis** - Caching and real-time features
5. **Scheduler** - Automated model retraining
6. **Nginx** - Reverse proxy and load balancer

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+ (for K8s deployment)
- 8GB RAM minimum
- 50GB disk space

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ecommerce
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Prepare Data

Place your data files in the `data/` directory:
- olist_customers_dataset.csv
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_order_payments_dataset.csv
- olist_order_reviews_dataset.csv
- olist_products_dataset.csv
- olist_sellers_dataset.csv
- olist_geolocation_dataset.csv

### 4. Deploy with Docker Compose

```bash
# Development deployment
./deployment/deploy.sh development

# Production deployment
./deployment/deploy.sh production
```

## Docker Deployment

### Build Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build app
```

### Start Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d app

# View logs
docker-compose logs -f
```

### Scale Services

```bash
# Scale API service
docker-compose up -d --scale api=3
```

## Kubernetes Deployment

### 1. Build and Push Images

```bash
# Set your registry
export DOCKER_REGISTRY=your-registry.com

# Build and push
./deployment/deploy.sh production
```

### 2. Update Kubernetes Manifests

Edit `deployment/kubernetes/deployment.yaml`:
- Update image names with your registry
- Configure resource limits
- Set ingress hostname

### 3. Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check deployment status
kubectl get pods -n ecommerce-analytics

# Get service endpoints
kubectl get svc -n ecommerce-analytics
```

### 4. Configure Ingress

```bash
# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Update ingress with your domain
kubectl edit ingress ecommerce-ingress -n ecommerce-analytics
```

## Production Configuration

### 1. Security

#### API Keys
- Set strong API keys in `.env`
- Rotate keys regularly
- Use secrets management (K8s secrets, AWS Secrets Manager)

#### SSL/TLS
- Configure SSL certificates in Nginx
- Use cert-manager for automatic SSL in K8s

#### Authentication
- Enable basic auth for MLflow UI
- Configure JWT tokens for API

### 2. Performance Optimization

#### Caching
- Configure Redis memory limits
- Set appropriate TTL for cache entries

#### Resource Limits
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

#### Horizontal Scaling
- API service: 3-10 replicas
- App service: 2-5 replicas
- Configure HPA for auto-scaling

### 3. Monitoring

#### Health Checks
- Streamlit: `http://localhost:8501/_stcore/health`
- API: `http://localhost:8000/health`
- MLflow: `http://localhost:5000/health`

#### Metrics
- Prometheus endpoint: `http://localhost:8000/metrics`
- Configure Grafana dashboards

#### Logging
- Centralized logging with ELK stack
- Log aggregation with Fluentd

## Model Management

### Initial Model Training

```bash
# Run training pipeline
docker exec -it ecommerce-analytics-app python pipeline/final_pipeline.py
```

### Automated Retraining

The scheduler service automatically retrains models based on the schedule in `.env`:
```
RETRAIN_SCHEDULE=0 2 * * 0  # Weekly on Sundays at 2 AM
```

### Manual Retraining

```bash
# Trigger manual retraining
docker exec -it ecommerce-analytics-scheduler python scheduler/retrain_scheduler.py
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Churn Prediction
```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
      "customer_id": "123",
      "recency_days": 30,
      "frequency": 5,
      "monetary_value": 500.0,
      "customer_lifetime_days": 365,
      "total_orders": 10,
      "avg_order_value": 50.0
    }]
  }'
```

### Batch Predictions
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [...],
    "include_churn": true,
    "include_segmentation": true,
    "include_recommendations": true
  }'
```

## Marketing Integration

### Configure Integrations

Set API keys in `.env`:
```bash
HUBSPOT_API_KEY=your_key
MAILCHIMP_API_KEY=your_key
MAILCHIMP_SERVER_PREFIX=us1
```

### Sync Data

The platform automatically syncs:
- Customer segments
- Churn predictions
- Campaign triggers

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   docker-compose logs app
   # Check for missing dependencies or data files
   ```

2. **API returns 503**
   ```bash
   # Check if models are loaded
   curl http://localhost:8000/model/info
   ```

3. **Out of memory**
   ```bash
   # Increase Docker memory limit
   docker update --memory="4g" container_name
   ```

### Debug Mode

```bash
# Run in debug mode
DEBUG=true docker-compose up
```

### Reset Everything

```bash
# Stop and remove all containers
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Fresh start
./deployment/deploy.sh development
```

## Maintenance

### Backup

```bash
# Backup models and artifacts
tar -czf backup_$(date +%Y%m%d).tar.gz artifacts/ mlruns/

# Backup to S3
aws s3 cp backup_*.tar.gz s3://your-bucket/backups/
```

### Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and deploy
./deployment/deploy.sh production
```

### Database Migrations

```bash
# Run migrations (if using database)
docker exec -it ecommerce-analytics-app python manage.py migrate
```

## Security Best Practices

1. **Secrets Management**
   - Never commit secrets to git
   - Use environment variables
   - Rotate keys regularly

2. **Network Security**
   - Use internal networks for service communication
   - Expose only necessary ports
   - Configure firewall rules

3. **Data Security**
   - Encrypt data at rest
   - Use SSL for all communications
   - Implement access controls

## Support

For issues or questions:

1. Check logs: `docker-compose logs -f`
2. Review documentation
3. Contact support team

## Appendix

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| APP_ENV | Environment (development/production) | production |
| API_WORKERS | Number of API workers | 4 |
| REDIS_HOST | Redis hostname | redis |
| MLFLOW_TRACKING_URI | MLflow tracking server | http://mlflow:5000 |
| RETRAIN_SCHEDULE | Cron schedule for retraining | 0 2 * * 0 |

### Port Mapping

| Service | Internal Port | External Port |
|---------|--------------|---------------|
| App | 8501 | 8501 |
| API | 8000 | 8000 |
| MLflow | 5000 | 5000 |
| Redis | 6379 | 6379 |
| Nginx | 80 | 80 |