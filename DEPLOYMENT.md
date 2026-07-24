# Deployment Guide

This guide covers deploying the Academic Citation Platform using Docker and Google Cloud Run.

## Table of Contents
- [Local Development with Docker](#local-development-with-docker)
- [Cloud Run Deployment](#cloud-run-deployment)
- [Environment Configuration](#environment-configuration)
- [Troubleshooting](#troubleshooting)

---

## Local Development with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available for containers

### Quick Start

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Access the application:**
   - Streamlit app: http://localhost:8080
   - Neo4j browser: http://localhost:7474 (neo4j/password123)

3. **View logs:**
   ```bash
   docker-compose logs -f app
   docker-compose logs -f neo4j
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

5. **Clean restart (removes data):**
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Configuration

Edit `docker-compose.yml` to customize:
- Neo4j credentials (default: neo4j/password123)
- Memory allocation
- Port mappings
- Volume mounts

---

## Cloud Run Deployment

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Project ID from GCP Console

### Setup Secrets

Before deploying, create secrets in Google Cloud Secret Manager:

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create secrets for Neo4j connection
echo -n "neo4j+s://your-aura-instance.databases.neo4j.io" | gcloud secrets create NEO4J_URI --data-file=-
echo -n "neo4j" | gcloud secrets create NEO4J_USER --data-file=-
echo -n "your-password" | gcloud secrets create NEO4J_PASSWORD --data-file=-
```

**Note:** For production, use [Neo4j Aura](https://neo4j.com/cloud/aura/) (free tier available) or host Neo4j separately.

### Deploy

1. **Set environment variables (optional):**
   ```bash
   export PROJECT_ID="your-gcp-project-id"
   export REGION="us-central1"
   export SERVICE="academic-citation-platform"
   ```

2. **Run deployment script:**
   ```bash
   ./deploy_cloud_run_container.sh
   ```

3. **Wait for deployment:**
   - Cloud Build will build the Docker image (~5-10 minutes first time)
   - Cloud Run will deploy the service
   - You'll receive a URL when complete

### Deployment Configuration

The deployment script configures:
- **CPU:** 2 vCPUs
- **Memory:** 2GB RAM
- **Concurrency:** 10 requests per instance
- **Scaling:** 0-3 instances (scales to zero when idle)
- **Timeout:** 300 seconds
- **Authentication:** Public (unauthenticated access)

To modify these, edit `deploy_cloud_run_container.sh` line 60-67.

### Cost Optimization

Cloud Run pricing is pay-per-use:
- Free tier: 2 million requests/month
- Charges apply when over free tier
- Scales to zero = no charges when idle

**Tips:**
- Keep `min-instances 0` to avoid idle costs
- Monitor usage in GCP Console
- Set billing alerts

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j database connection string | `neo4j+s://xxx.databases.neo4j.io` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `your-secure-password` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Streamlit server port | `8080` |

---

## Neo4j Setup Options

### Option 1: Neo4j Aura (Recommended for Cloud Run)
1. Sign up at https://neo4j.com/cloud/aura/
2. Create free AuraDB instance
3. Note connection URI, username, password
4. Add to Secret Manager (see Setup Secrets above)

### Option 2: Self-hosted Neo4j
- Use docker-compose for local development
- For production, deploy Neo4j on separate server/VM
- Point Cloud Run to Neo4j's public endpoint

### Option 3: Neo4j on Google Cloud
- Deploy Neo4j using GCP Marketplace
- Use VPC connector for private networking
- More complex but better security

---

## Troubleshooting

### Local Docker Issues

**App won't start:**
```bash
# Check logs
docker-compose logs app

# Rebuild image
docker-compose up --build
```

**Neo4j connection failed:**
```bash
# Verify Neo4j is healthy
docker-compose ps

# Check Neo4j logs
docker-compose logs neo4j

# Wait for Neo4j to fully start (can take 30-60 seconds)
```

**Out of memory:**
```bash
# Increase Docker Desktop memory allocation
# Settings → Resources → Memory (recommend 6GB+)
```

### Cloud Run Issues

**Build fails:**
- Check Cloud Build logs in GCP Console
- Verify `requirements.txt` is valid
- Ensure sufficient quota

**Service won't start:**
```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=academic-citation-platform" --limit 50
```

**Neo4j connection timeout:**
- Verify Secret Manager secrets are correct
- Check Neo4j Aura is running
- Ensure firewall allows Cloud Run IPs

**High costs:**
- Check `min-instances` is set to 0
- Review request logs for unexpected traffic
- Consider adding authentication

### Memory Issues on Free EC2

The Docker approach solves EC2 memory problems:
- Build happens in Cloud Build (not on EC2)
- Pre-built image is lightweight to pull
- No pip install on limited hardware

If still using EC2:
1. Add swap space (temporary fix)
2. Use smaller instance for deployment only
3. Build Docker image elsewhere, push to registry

---

## Next Steps

After successful deployment:

1. **Initialize database:**
   - Run `setup_database.py` to create schema
   - Upload demo data via Streamlit interface

2. **Monitor application:**
   - Cloud Run metrics in GCP Console
   - Logs in Cloud Logging
   - Set up uptime checks

3. **Customize:**
   - Update Streamlit theme/branding
   - Add authentication if needed
   - Configure custom domain

4. **Scale:**
   - Adjust `max-instances` based on traffic
   - Enable Cloud CDN for static assets
   - Implement caching strategy

---

## Support

For issues:
- Check Cloud Run logs: GCP Console → Cloud Run → Logs
- Verify secrets: GCP Console → Secret Manager
- Test locally first: `docker-compose up`

For questions about this deployment setup, refer to the main README or open an issue.
