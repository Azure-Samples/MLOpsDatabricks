import sys
import os
import time
import argparse

from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import DatabricksStep

sys.path.append(os.path.abspath("./aml_service/experiment"))
from workspace import get_workspace
from attach_compute import get_compute


def get_experiment_run_url(
    subscription_id,
    resource_group,
    aml_workspace_name,
    run_id
):
    base_url = 'https://mlworkspacecanary.azure.ai/portal/'
    base_url += 'subscriptions/%s/resourceGroups/%s/providers/'
    base_url += 'Microsoft.MachineLearningServices/workspaces/%s/'
    base_url += 'experiments/pipetest/runs/%s'

    formatted_url = base_url % (
        subscription_id,
        resource_group,
        aml_workspace_name,
        run_id)

    return formatted_url


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--cluster-id',
        required=True,
        help='Databricks Cluster id')
    arg_parser.add_argument(
        '--workspace-name',
        required=True,
        help='Azure ML Service Workspace name')
    arg_parser.add_argument(
        '--resource-group',
        required=True,
        help='Resource Group containing the existing Databricks Workspace')
    arg_parser.add_argument(
        '--subscription-id',
        required=True,
        help='ID of the Azure Subscription')
    arg_parser.add_argument(
        '--tenant-id',
        required=True,
        help='Azure Tenant ID')
    arg_parser.add_argument(
        '--app-id',
        required=True,
        help='Service Principal Application ID')
    arg_parser.add_argument(
        '--app-secret',
        required=True,
        help='Service Principal Application Secret (password)')
    arg_parser.add_argument(
        '--experiment-folder',
        required=True,
        help='Folder that stores the experiment files')
    arg_parser.add_argument(
        '--project-folder',
        required=True,
        help='Root folder for the whole project')
    arg_parser.add_argument(
        '--train-script-path',
        required=True,
        help='Train script path, relative to --project-folder')
    arg_parser.add_argument(
        '--databricks-workspace',
        required=True,
        help='Databricks Workspace name')
    arg_parser.add_argument(
        '--databricks-access-token',
        required=True,
        help='Databricks Access Token')
    arg_parser.add_argument(
        '--databricks-compute-name',
        required=True,
        help='Databricks Compute Name (at AML WS)')
    arg_parser.add_argument(
        '--model-dir',
        required=True,
        help='The Model Path')
    arg_parser.add_argument(
        '--model-name',
        required=True,
        help='The Model Name')

    main_arguments = arg_parser.parse_args()

    model_file_name = "%s.pth" % (main_arguments.model_name)
    model_path = os.path.join(main_arguments.model_dir, model_file_name)

    print("The model path will be %s" % (model_path))

    aml_workspace = get_workspace(
        main_arguments.workspace_name,
        main_arguments.resource_group,
        main_arguments.subscription_id,
        main_arguments.tenant_id,
        main_arguments.app_id,
        main_arguments.app_secret)
    print(aml_workspace)

    databricks_compute = get_compute(
        aml_workspace,
        main_arguments.databricks_compute_name,
        main_arguments.resource_group,
        main_arguments.databricks_workspace,
        main_arguments.databricks_access_token)
    print(databricks_compute)

    step1 = DatabricksStep(
        name="DBPythonInLocalMachine",
        num_workers=1,
        python_script_name=main_arguments.train_script_path,
        source_directory=main_arguments.project_folder,
        run_name='DB_Python_Local_demo',
        existing_cluster_id=main_arguments.cluster_id,
        compute_target=databricks_compute,
        allow_reuse=False,
        python_script_params=['--MODEL_PATH', model_path]
    )

    step2 = DatabricksStep(
        name="RegisterModel",
        num_workers=1,
        python_script_name="register_model.py",
        source_directory=main_arguments.experiment_folder,
        run_name='Register_model',
        existing_cluster_id=main_arguments.cluster_id,
        compute_target=databricks_compute,
        allow_reuse=False,
        python_script_params=[
            '--MODEL_PATH', model_path,
            '--TENANT_ID', main_arguments.tenant_id,
            '--APP_ID', main_arguments.app_id,
            '--APP_SECRET', main_arguments.app_secret,
            '--MODEL_NAME', main_arguments.model_name]
    )

    step2.run_after(step1)
    print("Step lists created")

    pipeline = Pipeline(
        workspace=aml_workspace,
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
        main_arguments.subscription_id,
        main_arguments.resource_group,
        main_arguments.workspace_name,
        pipeline_run_id
    )

    print("To check details of the Pipeline run, go to " + azure_run_url)

    pipeline_status = pipeline_run.get_status()

    timer_mod = 0

    while pipeline_status == 'Running':
        timer_mod = timer_mod + 10
        time.sleep(10)
        if (timer_mod % 30) == 0:
            print("Still running. %s seconds have passed." % (timer_mod))
        pipeline_status = pipeline_run.get_status()

    if pipeline_status == 'Failed':
        print("AML Pipelne failed. Check %s for details." % (azure_run_url))
        sys.exit(1)
    else:
        print(pipeline_status)

    print("Pipeline completed")


if __name__ == '__main__':
    main()
