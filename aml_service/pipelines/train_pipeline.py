import sys
import os
import time
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import DatabricksStep
sys.path.append(os.path.abspath("./aml_service/experiment"))
from attach_compute import get_compute
from workspace import get_workspace


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
    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID", None)

    # If databricks_cluster_id is not None, but it's an empty string: its None
    if cluster_id is not None and not cluster_id:
        cluster_id = None

    workspace_name = os.environ.get("AML_WORKSPACE_NAME", None)
    resource_group = os.environ.get("RESOURCE_GROUP", None)
    subscription_id = os.environ.get("SUBSCRIPTION_ID", None)
    tenant_id = os.environ.get("TENANT_ID", None)
    app_id = os.environ.get("SP_APP_ID", None)
    app_secret = os.environ.get("SP_APP_SECRET", None)
    experiment_subfolder = os.environ.get(
        "EXPERIMENT_FOLDER",
        'aml_service/experiment'
    )
    sources_directory = os.environ.get("SOURCES_DIR", None)
    experiment_folder = os.path.join(sources_directory, experiment_subfolder)
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH", None)
    databricks_workspace_name = os.environ.get(
        "DATABRICKS_WORKSPACE_NAME",
        None
    )
    databricks_access_token = os.environ.get("DATABRICKS_ACCESS_TOKEN", None)
    databricks_compute_name_aml = os.environ.get(
        "DATABRICKS_COMPUTE_NAME_AML",
        None
    )
    model_dir = os.environ.get("MODEL_DIR", 'dbfs:/model')
    model_name = os.environ.get("MODEL_NAME", 'torchcnn')

    path_components = model_dir.split("/", 1)
    model_path = "/dbfs/" + path_components[1] + "/" + model_name + ".pth"

    print("The model path will be %s" % (model_path))

    aml_workspace = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)
    print(aml_workspace)

    databricks_compute = get_compute(
        aml_workspace,
        databricks_compute_name_aml,
        resource_group,
        databricks_workspace_name,
        databricks_access_token)
    print(databricks_compute)

    step1 = DatabricksStep(
        name="DBPythonInLocalMachine",
        num_workers=1,
        python_script_name=train_script_path,
        source_directory=sources_directory,
        run_name='DB_Python_Local_demo',
        existing_cluster_id=cluster_id,
        compute_target=databricks_compute,
        allow_reuse=False,
        python_script_params=['--MODEL_PATH', model_path]
    )

    step2 = DatabricksStep(
        name="RegisterModel",
        num_workers=1,
        python_script_name="register_model.py",
        source_directory=experiment_folder,
        run_name='Register_model',
        existing_cluster_id=cluster_id,
        compute_target=databricks_compute,
        allow_reuse=False,
        python_script_params=[
            '--MODEL_PATH', model_path,
            '--TENANT_ID', tenant_id,
            '--APP_ID', app_id,
            '--APP_SECRET', app_secret,
            '--MODEL_NAME', model_name]
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
        subscription_id,
        resource_group,
        workspace_name,
        pipeline_run_id
    )

    print("To check details of the Pipeline run, go to " + azure_run_url)

    pipeline_status = pipeline_run.get_status()

    timer_mod = 0

    while pipeline_status == 'Running' or pipeline_status == 'NotStarted':
        timer_mod = timer_mod + 10
        time.sleep(10)
        if (timer_mod % 30) == 0:
            print(
                "Status: %s. %s seconds have passed." %
                (pipeline_status, timer_mod)
            )
        pipeline_status = pipeline_run.get_status()

    if pipeline_status == 'Failed':
        print("AML Pipelne failed. Check %s for details." % (azure_run_url))
        sys.exit(1)
    else:
        print(pipeline_status)

    print("Pipeline completed")


if __name__ == '__main__':
    main()
