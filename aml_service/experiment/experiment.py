import argparse
from azureml.core import Experiment, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--AZUREML_ARM_SUBSCRIPTION')
PARSER.add_argument('--AZUREML_ARM_RESOURCEGROUP')
PARSER.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
PARSER.add_argument('--TENANT_ID')
PARSER.add_argument('--APP_ID')
PARSER.add_argument('--APP_SECRET')

ARGS = PARSER.parse_args()

WORKSPACE_NAME = ARGS.AZUREML_ARM_WORKSPACE_NAME
RESOURCE_GROUP = ARGS.AZUREML_ARM_RESOURCEGROUP
SUBSCRIPTION_ID = ARGS.AZUREML_ARM_SUBSCRIPTION
TENANT_ID = ARGS.TENANT_ID
APP_ID = ARGS.APP_ID
APP_SECRET = ARGS.APP_SECRET

SP_AUTH = ServicePrincipalAuthentication(
    tenant_id=TENANT_ID,
    service_principal_id=APP_ID,
    service_principal_password=APP_SECRET)

WORKSPACE = Workspace.get(
    WORKSPACE_NAME,
    SP_AUTH,
    SUBSCRIPTION_ID,
    RESOURCE_GROUP
)

EXPERIMENT = Experiment(workspace=WORKSPACE, name="trainpipeline")

print(EXPERIMENT.name, EXPERIMENT.workspace.name, sep='\n')

EXPERIMENT_RUN = EXPERIMENT.start_logging()

EXPERIMENT_RUN.log('my magic number', 45)

EXPERIMENT_RUN.complete()
