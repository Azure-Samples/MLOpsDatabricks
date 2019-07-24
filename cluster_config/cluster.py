import requests
import base64
import json


class DatabricksCluster:
    name = None
    id = None
    state = None
    databricks_domain = None
    python_libraries = []

    def __init__(
        self,
        id,
        suffix=None,
        databricks_domain="centralus.azuredatabricks.net"
    ):
        if suffix is None:
            self.name = 'aml-cluster'
        else:
            self.name = 'aml-cluster-%s' % (suffix)

        self.databricks_domain = databricks_domain

        self.id = id

    def get_base_url(self):
        base_url = 'https://%s/api/2.0/' % (self.databricks_domain)

        return base_url

    def create(
        self,
        databricks_token,
        vm_type="Standard_D3_v2"
    ):
        response = requests.post(
            self.get_base_url() + 'clusters/create',
            headers={
                'Authorization': b"Basic " + base64.standard_b64encode(
                    b"token:" + str.encode(databricks_token)
                )
            },
            json={
                "cluster_name": self.name,
                "spark_version": "5.4.x-cpu-ml-scala2.11",
                "node_type_id": vm_type,
                "driver_node_type_id": vm_type,
                "spark_env_vars": {
                    "PYSPARK_PYTHON": "/databricks/python3/bin/python3"
                },
                "autoscale": {
                    "min_workers": 1,
                    "max_workers": 2
                },
                "autotermination_minutes": 60
            })

        if response.status_code == 200:
            self.id = response.json()['cluster_id']

            self.get_state(databricks_token)
        else:
            raise ClusterManagementException(
                "Error creating the cluster: %s:%s" % (
                    response.json()["error_code"],
                    response.json()["message"]
                )
            )

    def get_state(self, databricks_token):
        base_url = self.get_base_url()

        response = requests.post(
            base_url + 'clusters/get',
            headers={
                'Authorization': b"Basic " + base64.standard_b64encode(
                    b"token:" + str.encode(databricks_token))},
            json={
                "cluster_id": self.id
            })

        if response.status_code == 200:
            self.state = response.json()['state']
        else:
            raise ClusterManagementException(
                "Error getting cluster state: %s: %s" %
                (response.json()["error_code"], response.json()["message"])
            )

    def start(self, databricks_token):
        if not self.id:
            raise ClusterManagementException(
                "The Cluster ID is undefined"
            )

        if self.state is None:
            self.get_state(databricks_token)

        if self.state == 'TERMINATED':
            response = requests.post(
                self.get_base_url() + 'clusters/start',
                headers={
                    'Authorization': b"Basic " + base64.standard_b64encode(
                        b"token:" + str.encode(databricks_token))
                },
                json={
                    "cluster_id": self.id
                })

            if response.status_code == 200:
                self.get_state(databricks_token)
            else:
                raise ClusterManagementException(
                    "Error starting cluster: %s: %s" %
                    (response.json()["error_code"], response.json()["message"])
                )

    def uninstall_libraries(self, databricks_token, library_config_file_path):
        if self.state is None:
            self.get_state(databricks_token)

        if self.state == 'TERMINATED':
            return

        with open(library_config_file_path, "r") as content:
            libraries = json.load(content)

        response = requests.post(
            self.get_base_url() + 'libraries/uninstall',
            headers={
                'Authorization': b"Basic " + base64.standard_b64encode(
                    b"token:" + str.encode(databricks_token))
            },
            json={
                "cluster_id": self.id,
                "libraries": libraries
            })

        if response.status_code != 200:
            raise ClusterManagementException(
                "Error uninstalling libraries: %s: %s" %
                (response.json()["error_code"], response.json()["message"])
            )

    def install_libraries(self, databricks_token, library_config_file_path):
        if self.state is None:
            self.get_state(databricks_token)

        if self.state != 'RUNNING':
            raise ClusterManagementException(
                "The cluster %s is not RUNNING (%s)" %
                (self.id, self.state)
            )

        with open(library_config_file_path, "r") as content:
            libraries = json.load(content)

        response = requests.post(
            self.get_base_url() + 'libraries/install',
            headers={
                'Authorization': b"Basic " + base64.standard_b64encode(
                    b"token:" + str.encode(databricks_token))
            },
            json={
                "cluster_id": self.id,
                "libraries": libraries
            })

        if response.status_code != 200:
            raise ClusterManagementException(
                "Error installing libraries: %s: %s" %
                (response.json()["error_code"], response.json()["message"])
            )

    def check_libraries(self, databricks_token):
        response = requests.get(
            self.get_base_url() + 'libraries/cluster-status',
            headers={
                'Authorization': b"Basic " + base64.standard_b64encode(
                    b"token:" + str.encode(databricks_token))
            },
            json={
                "cluster_id": self.id
            })

        if response.status_code == 200:
            for library in response.json()["library_statuses"]:
                if library["status"] == "INSTALLING":
                    return "INSTALLING"

            return "INSTALLED"
        else:
            raise ClusterManagementException(
                "Error installing libraries: %s: %s" %
                (
                    response.json()["error_code"],
                    response.json()["message"]
                )
            )

    def terminate(
        self,
        databricks_token,
        library_config_file_path,
        permanent=False
    ):
        if permanent is True:
            api_url = 'clusters/permanent-delete'
        else:
            api_url = 'clusters/delete'
            self.uninstall_libraries(
                databricks_token,
                library_config_file_path
            )
            print(
                "Requested to uninstall libraries from %s" %
                (self.id)
            )

        response = requests.post(
            self.get_base_url() + api_url,
            headers={
                'Authorization': b"Basic " + base64.standard_b64encode(
                    b"token:" + str.encode(databricks_token))
            },
            json={
                "cluster_id": self.id
            }),

        if response[0].status_code != 200:
            raise ClusterManagementException(
                "Error terminating cluster: %s: %s" %
                (
                    response[0].json()["error_code"],
                    response[0].json()["message"]
                )
            )


class ClusterManagementException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
