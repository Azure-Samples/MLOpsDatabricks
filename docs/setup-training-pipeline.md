# Setting up the Training Pipeline

#TODO

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