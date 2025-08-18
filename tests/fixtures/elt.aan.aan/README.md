# ELT AAN (Account Analysis and Notification)

This project implements ETL processes for account analysis and notification systems.

## Overview

The AAN ETL pipeline processes customer account data and generates insights for:
- Account behavior analysis
- Risk assessment notifications
- Data quality monitoring
- Performance metrics generation

## Structure

- `src/main/scala/` - Scala source code for ETL jobs
- `src/main/run/config/` - Environment-specific configuration files
- `src/main/houston/` - Deployment scripts for Houston platform
- `project/` - SBT build configuration

## Configuration

Environment-specific configurations are available in `src/main/run/config/`:
- `datascience.dev.conf` - Development environment
- `datascience.test.conf` - Test environment  
- `datascience.preprod.conf` - Pre-production environment
- `scoring.prod.conf` - Production scoring configuration

## Building

```bash
sbt clean compile
sbt assembly
```

## Testing

```bash
sbt test
```

## Deployment

Use Houston deployment scripts in `src/main/houston/` for automated deployment.