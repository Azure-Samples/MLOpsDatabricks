import sys
import os
import time
import click
import json

from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import DatabricksStep

sys.path.append(os.path.abspath("./aml_service/experiment"))
from AttachCompute import get_compute
from WorkSpace import get_workspace


def get_experiment_run_url(
    subscription_id,
    resource_group,
    aml_workspace_name,
    run_id
):
    base_url = 'https://mlworkspacecanary.azure.ai/portal/subscriptions/%s/resourceGroups/%s/providers/Microsoft.MachineLearningServices/workspaces/%s/experiments/pipetest/runs/%s'
    formatted_url = base_url % (
        subscription_id,
        resource_group,
        aml_workspace_name,
        run_id)

    return formatted_url


@click.command()
@click.option('--cluster-id', required=True, help='Databricks Cluster id') 
@click.option('--workspace-name', required=True, help='Azure ML Service Workspace name')
@click.option('--resource-group', required=True, help='Resource Group containing the existing Databricks Workspace')
@click.option('--subscription-id', required=True, help='ID of the Azure Subscription')
@click.option('--tenant-id', required=True, help='Azure Tenant ID')
@click.option('--app-id', required=True, help='Service Principal Application ID')
@click.option('--app-secret', required=True, help='Service Principal Application Secret (password)')
@click.option('--experiment-folder', required=True, help='Folder that stores the experiment files')
@click.option('--project-folder', required=True, help='Root folder for the whole project')
@click.option('--train-script-path', required=True, help='Train script path, relative to --project-folder')
@click.option('--databricks-workspace', required=True, help='Databricks Workspace name')
@click.option('--databricks-access-token', required=True, help='Databricks Access Token')
@click.option('--databricks-compute-name', required=True, help='Databricks Compute Name (at AML WS)')
@click.option('--mdl-dir', required=True, help='The Model Path')
@click.option('--mdl-name', required=True, help='The Model Name')
def main(
    cluster_id,
    workspace_name,
    resource_group,
    subscription_id,
    tenant_id,
    app_id,
    app_secret,
    experiment_folder,
    project_folder,
    train_script_path,
    databricks_workspace,
    databricks_access_token,
    databricks_compute_name,
    mdl_dir,
    mdl_name
):
    mdl_file_name = "%s.pth" % (mdl_name)
    mdl_path = os.path.join(mdl_dir, mdl_file_name)

    print("The model path will be %s" % (mdl_path))

    ws = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)
    print(ws)

    databricks_compute = get_compute(
        ws,
        databricks_compute_name,
        resource_group,
        databricks_workspace,
        databricks_access_token)
    print(databricks_compute)

    step1 = DatabricksStep(
        name="DBPythonInLocalMachine",
        num_workers=1,
        python_script_name=train_script_path,
        source_directory=project_folder,
        run_name='DB_Python_Local_demo',
        existing_cluster_id=cluster_id,
        compute_target=databricks_compute,
        allow_reuse=False,
        python_script_params=['--MODEL_PATH', mdl_path]
    )

    step2 = DatabricksStep(
        name="RegisterModel",
        num_workers=1,
        python_script_name="Register.py",
        source_directory=experiment_folder,
        run_name='Register_model',
        existing_cluster_id=cluster_id,
        compute_target=databricks_compute,
        allow_reuse=False,
        python_script_params=[
            '--MODEL_PATH', mdl_path,
            '--TENANT_ID', tenant_id,
            '--APP_ID', app_id,
            '--APP_SECRET', app_secret,
            '--MODEL_NAME', mdl_name]
    )

    step2.run_after(step1)
    print("Step lists created")

    pipeline = Pipeline(
        workspace=ws,
        # steps=[step1])
        steps=[step1, step2])
    print("Pipeline is built")

    pipeline.validate()
    print("Pipeline validation complete")

    pipeline_run = pipeline.submit(experiment_name="pipetest")

    print("Pipeline is submitted for execution")

    pipeline_details = pipeline_run.get_details()

    pipeline_run_id = pipeline_details['runId']

    azure_run_url = get_experiment_run_url(
        subscription_id,
        resource_group,
        workspace_name,
        pipeline_run_id
    )

    print("To check details of the Pipeline run, go to " + azure_run_url)

    pipeline_status = pipeline_run.get_status()

    timer_mod = 0

    while(pipeline_status == 'Running'):
        timer_mod = timer_mod + 10
        time.sleep(10)
        if((timer_mod % 30) == 0):
            print("Still running. %s seconds have passed." % (timer_mod))
        pipeline_status = pipeline_run.get_status()

    if pipeline_status == 'Failed':
        sys.exit("AML Pipeline failed")
    else:
        print(pipeline_status)

    print("Pipeline completed")


if __name__ == '__main__':
    main()
