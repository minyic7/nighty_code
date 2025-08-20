#!/bin/bash

# Houston Deployment Script for ELT AAN
# This script handles deployment of the AAN ETL jobs to the Houston platform

set -euo pipefail

# Configuration
PROJECT_NAME="elt-aan-aan"
ARTIFACT_PATH="target/scala-2.12/${PROJECT_NAME}-assembly-1.0.0.jar"
HOUSTON_ENDPOINT="${HOUSTON_ENDPOINT:-https://houston.internal.company.com}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Validate prerequisites
validate_environment() {
    log "Validating deployment environment..."
    
    if [ ! -f "${ARTIFACT_PATH}" ]; then
        error "Assembly JAR not found at ${ARTIFACT_PATH}. Run 'sbt assembly' first."
    fi
    
    if [ -z "${HOUSTON_TOKEN:-}" ]; then
        error "HOUSTON_TOKEN environment variable is not set"
    fi
    
    log "Environment validation completed"
}

# Deploy to Houston
deploy_to_houston() {
    log "Starting deployment to Houston (${ENVIRONMENT})..."
    
    # Upload artifact
    log "Uploading artifact..."
    curl -X POST \
        -H "Authorization: Bearer ${HOUSTON_TOKEN}" \
        -H "Content-Type: application/octet-stream" \
        -T "${ARTIFACT_PATH}" \
        "${HOUSTON_ENDPOINT}/api/v1/projects/${PROJECT_NAME}/artifacts"
    
    # Trigger deployment
    log "Triggering deployment..."
    curl -X POST \
        -H "Authorization: Bearer ${HOUSTON_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"environment\":\"${ENVIRONMENT}\",\"version\":\"1.0.0\"}" \
        "${HOUSTON_ENDPOINT}/api/v1/projects/${PROJECT_NAME}/deploy"
    
    log "Deployment request submitted successfully"
}

# Main execution
main() {
    log "Starting Houston deployment for ${PROJECT_NAME}"
    validate_environment
    deploy_to_houston
    log "Deployment process completed"
}

# Execute main function
main "$@"