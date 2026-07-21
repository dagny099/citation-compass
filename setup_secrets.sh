#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project ID (e.g. export PROJECT_ID=my-gcp-project)}"

echo "Setting up secrets for Academic Citation Platform in project: $PROJECT_ID"
echo ""

# Set project
gcloud config set project "$PROJECT_ID"

# Enable Secret Manager API
echo "Enabling Secret Manager API..."
gcloud services enable secretmanager.googleapis.com

echo ""
echo "=========================================="
echo "Creating secrets in Secret Manager"
echo "=========================================="
echo ""

# Prompt for Neo4j URI
read -p "Enter Neo4j URI (e.g., neo4j+s://xxxxx.databases.neo4j.io or bolt://your-server:7687): " NEO4J_URI_VALUE

# Create or update NEO4J_URI secret
if gcloud secrets describe NEO4J_URI >/dev/null 2>&1; then
  echo "Secret NEO4J_URI already exists, adding new version..."
  echo -n "$NEO4J_URI_VALUE" | gcloud secrets versions add NEO4J_URI --data-file=-
else
  echo "Creating secret NEO4J_URI..."
  echo -n "$NEO4J_URI_VALUE" | gcloud secrets create NEO4J_URI --data-file=-
fi

echo "✅ NEO4J_URI secret created/updated"
echo ""

# Prompt for Neo4j user
read -p "Enter Neo4j username [neo4j]: " NEO4J_USER_VALUE
NEO4J_USER_VALUE=${NEO4J_USER_VALUE:-neo4j}

# Create or update NEO4J_USER secret
if gcloud secrets describe NEO4J_USER >/dev/null 2>&1; then
  echo "Secret NEO4J_USER already exists, adding new version..."
  echo -n "$NEO4J_USER_VALUE" | gcloud secrets versions add NEO4J_USER --data-file=-
else
  echo "Creating secret NEO4J_USER..."
  echo -n "$NEO4J_USER_VALUE" | gcloud secrets create NEO4J_USER --data-file=-
fi

echo "✅ NEO4J_USER secret created/updated"
echo ""

# Prompt for Neo4j password
read -sp "Enter Neo4j password: " NEO4J_PASSWORD_VALUE
echo ""

# Create or update NEO4J_PASSWORD secret
if gcloud secrets describe NEO4J_PASSWORD >/dev/null 2>&1; then
  echo "Secret NEO4J_PASSWORD already exists, adding new version..."
  echo -n "$NEO4J_PASSWORD_VALUE" | gcloud secrets versions add NEO4J_PASSWORD --data-file=-
else
  echo "Creating secret NEO4J_PASSWORD..."
  echo -n "$NEO4J_PASSWORD_VALUE" | gcloud secrets create NEO4J_PASSWORD --data-file=-
fi

echo "✅ NEO4J_PASSWORD secret created/updated"
echo ""

echo "=========================================="
echo "Verifying secrets..."
echo "=========================================="
gcloud secrets list --filter="name:(NEO4J_URI OR NEO4J_USER OR NEO4J_PASSWORD)"

echo ""
echo "✅ All secrets configured successfully!"
echo ""
echo "You can now run: ./deploy_cloud_run_container.sh"
