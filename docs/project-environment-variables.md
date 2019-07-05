# Project environment variables

To run either locally or on Azure DevOps, this set of scripts need a few
environment variables to be set. 

This document lists and describes each one of them.

##  Cluster Management Environment Variables

### DATABRICKS_DOMAIN

The Databricks [instance name](https://docs.azuredatabricks.net/user-guide/faq/workspace-details.html#id1).

### DATABRICKS_ACCESS_TOKEN

The Databricks
[Personal Access Token](https://docs.azuredatabricks.net/api/latest/authentication.html#generate-a-token)
you have generated on your workspace.

### DATABRICKS_CLUSTER_NAME_SUFFIX

Optional. A suffix you can use to identify all of your clusters.
If you choose not to set one, all of your clusters will be called `aml-cluster`.


### DATABRICKS_CLUSTER_ID

Use this variable if you prefer to use an existing cluster to run your
train pipeline. Here, you can set the chosen `cluster id`. Check
[this doc](https://docs.azuredatabricks.net/user-guide/faq/workspace-details.html#cluster-url)
for more information about cluster ids.

## Train Environment Variables

### AML_WORKSPACE_NAME

The name of your Azure Machine Learning Service Workspace. Amongst all the 
generated infrastructure resources, this will have the `-AML-WS` suffix.

### RESOURCE_GROUP

The name of the Azure Resource Group that contains all of your workloads
(Databricks Workspace, Azure ML Service, etc.). Amongst all the 
generated infrastructure resources, this will have the `-AML-RG` suffix.

### SUBSCRIPTION_ID

The ID of the Azure Subscription that is being used to run this infrastructure.

### TENANT_ID

The ID of the Azure Active Directory Tenant
associated with the Azure Subscription

### SP_APP_ID

The Application ID of the Service Principal
[you have created](setup-training-pipeline.md#create-the-service-principal-for-the-azure-machine-learning-workspace)
to run this code.

### SP_APP_SECRET

The Application Secret (or password) of the Service Principal
[you have created](setup-training-pipeline.md#create-the-service-principal-for-the-azure-machine-learning-workspace)
to run this code.

### SOURCES_DIR

The root folder of this code.

#### On a local development environment

For example: if you have cloned this repository on `/home/username/projects`
and you have not customized the name of the repository local folder, the
`SOURCES_DIR` would be `/home/username/projects/MLOpsDatabricks`.

#### On Azure DevOps

If you're associating this repo with an Azure DevOps pipeline without
much customization to clone the repo, this variable can set as
`$(Build.SourcesDirectory)`.

### TRAIN_SCRIPT_PATH

If you use this code structure *as is*, the current train script path
is `src/train/train.py`.

### DATABRICKS_WORKSPACE_NAME

The name that was given to your Azure Databricks Workspace. Amongst all the 
generated infrastructure resources, this will have the `-AML-ADB` suffix.

### DATABRICKS_COMPUTE_NAME_AML

The name you want to give to the compute target that will be associated with the
Azure ML Service Workspace. This will attach the Databricks Cluster
to the Azure ML Service Workspace.

### MODEL_DIR

The directory to save the trained model to, on the Databricks Cluster.
It's recommended to use `/dbfs/model`.

### MODEL_NAME

The name to give to the trained model. For example: `my-trained-model`.

## Read more

* [What is an Azure Machine Learning service workspace?](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-workspace)
* [Getting your Azure Subscription GUID (new portal)](https://blogs.msdn.microsoft.com/mschray/2016/03/18/getting-your-azure-subscription-guid-new-portal/)
* [Use an existing (Azure) tenant - Getting the Azure Tenant ID](https://docs.microsoft.com/en-us/azure/active-directory/develop/quickstart-create-new-tenant#use-an-existing-tenant)
* [Azure Databricks Workspaces](https://docs.azuredatabricks.net/user-guide/workspace.html#workspace)
* [Azure Databricks as a compute target for Azure ML Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-create-your-first-pipeline#databricks)