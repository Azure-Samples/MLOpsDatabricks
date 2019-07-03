# Setting up the Training Pipeline

This Azure DevOps pipeline will help you operationalize the process to:
- Create or reuse a Databricks cluster to serve as a remote
compute to Azure ML Service
- Attach this cluster as a Compute to Azure ML Service
- Execute the Azure ML pipeline code on this cluster
- Terminate the cluster after the job is done

## Create the Service Principal for the Azure Machine Learning Workspace

```bash
az ad sp create-for-rbac \
  --name "<name-for-service-principal>" \
  --scopes "<Resource Group ID>"
```

Where:

- `<name-for-service-principal>` is any name with no
spaces and special characters
- `<Resource Group ID>` can be retrieved from the **Properties** tab of the
Resource Group blade that contains all your resources:

![Resource Group ID](images/training/02-resource-group-id.png)

## Import this pipeline to your AzDO account

You will use `azdo_pipelines\iac-pipeline-arm.yml` to create a new Azure
Pipeline to do all the work to train your model. All the steps are defined
on this file. To use this file to create the Train Pipeline on your account:

- On Azure DevOps, go to `Pipelines > Build`
  - `https://dev.azure.com/<azdo-tenant>/<team-project>/_build` where:
    - `your-azdo-tenant` is your Azure DevOps tenant containing all
    the team projects
    - `team project` is the Team Project you are using to run this sample
- Click `New > New Build Pipeline`
  - `https://dev.azure.com/<azdo-tenant>/<team-project>/_apps/hub/ms.vss-build-web.ci-designer-hub`
- Choose where is your code

