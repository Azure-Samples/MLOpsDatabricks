#!/bin/sh
# Cluster Management Environment Variables
export DATABRICKS_DOMAIN=""
export DATABRICKS_ACCESS_TOKEN=""
export DATABRICKS_CLUSTER_NAME_SUFFIX="mySuffix"
export DATABRICKS_CLUSTER_ID=""

# Train Environment Variables
export AML_WORKSPACE_NAME="MLOpsOSS-AML-WS"
export RESOURCE_GROUP="MLOpsOSS-AML-RG"
export SUBSCRIPTION_ID=""
export TENANT_ID=""
export SP_APP_ID=""
export SP_APP_SECRET=""
export SOURCES_DIR=""
export TRAIN_SCRIPT_PATH="src/train/train.py"
export DATABRICKS_WORKSPACE_NAME="MLOpsOSS-AML-ADB"
export DATABRICKS_COMPUTE_NAME_AML="ADB-Compute"
export MODEL_DIR="/dbfs/model"
export MODEL_NAME="mymodel"