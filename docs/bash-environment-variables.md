# Using bash to set the environment variables

If you're using any other IDE to work on this code other than
[Visual Studio Code](vscode-launch-json.md), we have provided
a *shell script* that you can use to have all the required
environment variables set so you can run and debug this code.

Once you clone this repo, the shell script will be located at
`./set-environment-vars.sh`. Before running it, make sure to fill in all
the variable values. The piece of code below is a sample of the script:

```bash
#!/bin/sh
# Cluster Management Environment Variables
export DATABRICKS_DOMAIN="eastus.azuredatabricks.net"
export DATABRICKS_ACCESS_TOKEN="my-adb-access-token"
export DATABRICKS_CLUSTER_NAME_SUFFIX="my-env"
export DATABRICKS_CLUSTER_ID="my-cluster-id"
```

Where:

* `eastus.azuredatabricks.net` is the
[instance name](https://docs.azuredatabricks.net/user-guide/faq/workspace-details.html#workspace-instance-and-id)
of your Azure Databricks Workspace
* `my-adb-access-token` is the Personal Access Token that
[you have generated](https://docs.azuredatabricks.net/api/latest/authentication.html#generate-a-token)
on your Azure Databricks Workspace
* `my-env` is a suffix you have chosen to give to every Cluster this
script creates at runtime
* `my-cluster-id` is the id of an existing Databricks Cluster,
if you prefer to use one instead of creating a new Cluster every run

To get the details about all the environment variables used on this sample,
check [Project environment variables](project-environment-variables.md).

## Running the script

On your shell terminal, navigate to the root folder of this repo and
run the following command:

```bash
source ./set-environment-vars.sh
```

To make sure the script worked, you can try printing out the values of
any variable that had the value set, for example:

```bash
echo $DATABRICKS_CLUSTER_NAME_SUFFIX
```

Should print:

```bash
my-env
```

## Read more

* [Generate a token - Azure Databricks](https://docs.azuredatabricks.net/api/latest/authentication.html#generate-a-token)
* [Workspace Instance and ID (talks about instance name)](https://docs.azuredatabricks.net/user-guide/faq/workspace-details.html#workspace-instance-and-id)
* [Cluster URL - Azure Databricks (talks about Cluster ID)](https://docs.azuredatabricks.net/user-guide/faq/workspace-details.html#cluster-url)