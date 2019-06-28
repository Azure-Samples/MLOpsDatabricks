import os, json
from azureml.core import Experiment, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

root_dir = os.path.abspath(__file__ + "/../../../")
script_dir = os.path.join(root_dir, "aml_config/config.json")

with open(script_dir) as f:
    config = json.load(f)

workspace_name = config['workspace_name']
resource_group = config['resource_group']
subscription_id = config['subscription_id']


ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group = resource_group)

exp = Experiment(workspace=ws, name="trainpipeline")

print(exp.name, exp.workspace.name, sep = '\n')

run = exp.start_logging()

run.log('my magic number', 45)

run.complete()