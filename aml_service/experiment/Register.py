import sys
import argparse
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication
import os.path


parser = argparse.ArgumentParser()
parser.add_argument('--AZUREML_RUN_TOKEN')
parser.add_argument('--AZUREML_RUN_ID')
parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')
parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
parser.add_argument('--MODEL_PATH')
parser.add_argument('--MODEL_NAME')
parser.add_argument('--TENANT_ID')
parser.add_argument('--APP_ID')
parser.add_argument('--APP_SECRET')

args = parser.parse_args()

tenant_id = args.TENANT_ID
app_id = args.APP_ID
app_secret = args.APP_SECRET
workspace_name = args.AZUREML_ARM_WORKSPACE_NAME
subscription_id = args.AZUREML_ARM_SUBSCRIPTION
resource_group = args.AZUREML_ARM_RESOURCEGROUP
mdl_path = args.MODEL_PATH
mdl_name = args.MODEL_NAME

if(os.path.isfile(mdl_path) is False):
    print("The given model path %s is invalid" % (mdl_path))
    sys.exit(1)

sp_auth = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=app_id,
    service_principal_password=app_secret)

ws = Workspace.get(
    name=workspace_name,
    auth=sp_auth,
    subscription_id=subscription_id,
    resource_group=resource_group
)

try:
    model = Model.register(
        model_path=mdl_path,
        model_name=mdl_name,
        description="Fashion MNIST",
        workspace=ws)

    print("Model registered successfully. ID: " + model.id)
except Exception as e:
    print("Error while registering the model: " + str(e))
    sys.exit(1)
