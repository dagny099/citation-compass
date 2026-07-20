#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gen-lang-client-0374286648}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-academic-citation-platform}"
DOMAIN="${DOMAIN:-cartography.barbhs.com}"

echo "Setting up custom domain: $DOMAIN for service: $SERVICE"
echo ""

# Set project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable domains.googleapis.com

echo ""
echo "=========================================="
echo "Step 1: Map domain to Cloud Run service"
echo "=========================================="
echo ""

# Add domain mapping
gcloud run domain-mappings create \
  --service "$SERVICE" \
  --domain "$DOMAIN" \
  --region "$REGION"

echo ""
echo "=========================================="
echo "Step 2: Get DNS records to configure"
echo "=========================================="
echo ""

# Get the DNS records needed
gcloud run domain-mappings describe "$DOMAIN" \
  --region "$REGION" \
  --format="table(
    status.resourceRecords[0].type,
    status.resourceRecords[0].name,
    status.resourceRecords[0].rrdata
  )"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Go to your DNS provider (where you manage barbhs.com)"
echo "2. Add the DNS records shown above"
echo "3. Wait for DNS propagation (can take 15min - 48hrs, usually ~1hr)"
echo "4. Verify with: gcloud run domain-mappings describe $DOMAIN --region $REGION"
echo ""
echo "You'll see an A record and possibly an AAAA record to add."
echo ""
