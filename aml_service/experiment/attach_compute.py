from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DatabricksCompute
from azureml.exceptions import ComputeTargetException


def get_compute(
    workspace: Workspace,
    dbcomputename: str,
    resource_group: str,
    dbworkspace: str,
    dbaccesstoken: str
):
    try:
        databricks_compute = DatabricksCompute(
            workspace=workspace,
            name=dbcomputename)
        print('Compute target {} already exists'.format(dbcomputename))
    except ComputeTargetException:
        print('Compute not found, will use below parameters to attach new one')
        print('db_compute_name {}'.format(dbcomputename))
        print('db_resource_group {}'.format(resource_group))
        print('db_workspace_name {}'.format(dbworkspace))

        config = DatabricksCompute.attach_configuration(
            resource_group=resource_group,
            workspace_name=dbworkspace,
            access_token=dbaccesstoken)

        databricks_compute = ComputeTarget.attach(
            workspace,
            dbcomputename,
            config)
        databricks_compute.wait_for_completion(True)
    return databricks_compute
