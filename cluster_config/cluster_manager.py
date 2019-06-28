import base64
import json
import click
import os
import requests
import sys
import time

DOMAIN = 'centralus.azuredatabricks.net'
TOKEN = b''
BASE_URL = 'https://%s/api/2.0/' % (DOMAIN)
CLUSTER_ID = None
CLUSTER_STATE = None
CLUSTER_NAME = None
CLUSTER_VMTYPE = "Standard_D3_v2"


def create_cluster():
    global CLUSTER_ID
    global CLUSTER_NAME
    global CLUSTER_VMTYPE

    print("Creating cluster of " + CLUSTER_VMTYPE + " type")

    if CLUSTER_ID is None:
        if CLUSTER_NAME is None:
            CLUSTER_NAME = 'aml-cluster'
        else:
            CLUSTER_NAME = "aml-cluster-" + CLUSTER_NAME
        # API call to create databricks cluster.
        response = requests.post(
            BASE_URL + 'clusters/create',
            headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
            json={
                "cluster_name": CLUSTER_NAME,
                "spark_version": "5.4.x-cpu-ml-scala2.11",
                "node_type_id": CLUSTER_VMTYPE,
                "driver_node_type_id": CLUSTER_VMTYPE,
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
            CLUSTER_ID = response.json()['cluster_id']
            print("Cluster created successfully: " + CLUSTER_ID)
        else:
            print("Error creating cluster: " + response.json()["error_code"] + ": " + response.json()["message"])
            sys.exit(1)


def start_cluster():
    # Check Cluster state
    cluster_state()

    print("Cluster state: " + CLUSTER_STATE)

    if(CLUSTER_STATE != 'TERMINATED'):
        print("Cluster " + CLUSTER_ID + " is not in TERMINATED state (ACTUAL: " + CLUSTER_STATE + "). Skipping...")
    else:
        # API call to start cluster.
        response = requests.post(
            BASE_URL + 'clusters/start',
            headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
            json={
                "cluster_id": CLUSTER_ID
            })

        if response.status_code == 200:
            print("Cluster Starting ..(but not yet ready)")
        else:
            print("Error starting cluster: %s: %s" % (response.json()["error_code"], response.json()["message"]))
            sys.exit(1)


def cluster_state():
    # Get cluster state.
    global CLUSTER_STATE
    response = requests.post(
        BASE_URL + 'clusters/get',
        headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
        json={
               "cluster_id": CLUSTER_ID
           })

    if response.status_code == 200:
        CLUSTER_STATE = response.json()['state']
    else:
        print("Error getting cluster state: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        sys.exit(1)


def install_libraries():
    # Ensure cluster is running before intalling packages 
    # https://docs.azuredatabricks.net/api/latest/clusters.html#clusterclusterstate

    cluster_state()

    if(CLUSTER_STATE == 'TERMINATED'):
        start_cluster()

    if(CLUSTER_STATE == 'PENDING'):
        while CLUSTER_STATE == "PENDING":
            time.sleep(5)
            cluster_state()
            print("Cluster state: " + CLUSTER_STATE)

    if(CLUSTER_STATE == 'RUNNING'):
        # Install Libraries
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'library.json')
        with open(file_path, "r") as content:
            libraries = json.load(content)

        response = requests.post(
            BASE_URL + 'libraries/install',
            headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
            json={
                "cluster_id": CLUSTER_ID,
                "libraries": libraries
            })

        if response.status_code == 200:
            print("Library installation started")
        else:
            print("Error installing libraries: %s: %s" % (response.json()["error_code"], response.json()["message"]))
            sys.exit(1)
    else:
        print("Error: The cluster " + CLUSTER_ID + " is on an invalid state: " + CLUSTER_STATE)
        sys.exit(1)


def check_libraries():
    response = requests.get(
        BASE_URL + 'libraries/cluster-status',
        headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
        json={
            "cluster_id": CLUSTER_ID
        }),

    if response[0].status_code == 200:
        library_statuses = response[0].json()['library_statuses']
        for status in library_statuses:
            print("Library : %s Status: %s" % (status['library'], status['status']))
    else:
        print("Error installing libraries: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        sys.exit(1)


def terminate_cluster():
    # API call to terminate cluster.
    response = requests.post(
        BASE_URL + 'clusters/delete',
        headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
        json={
               "cluster_id": CLUSTER_ID
           }),

    if response.status_code == 200:
        print("Cluster terminated sucessfully")
    else:
        print("Error terminating cluster")
        sys.exit(1)


def permanent_terminate_cluster():
    # API call to permanently delete cluster.
    response = requests.post(
        BASE_URL + 'clusters/permanent-delete',
        headers={'Authorization': b"Basic " + base64.standard_b64encode(b"token:" + TOKEN)},
        json={
               "cluster_id": CLUSTER_ID
           }),
    if response.status_code == 200:
        print("Cluster permanently terminated sucessfully")
    else:
        print("Error permanently terminating cluster")
        sys.exit(1)


@click.command()
@click.option('--domain', required=True, help='Domain of your databricks cluster e.g. centralus.azuredatabricks.net.') 
@click.option('--token', required=True, help='Access key.')
@click.option('--clustervmtype', help='Use this info to set cluster vm type/size.')
@click.option('--clusterid', help='Cluster ID.')
@click.option('--terminate', is_flag=True, default=False, help='Terminates cluster,requires --clusterid')
@click.option('--permanent', is_flag=True, default=False, help='Permanently deletes cluster, requires --clusterid and --terminate')
@click.option('--clustername', help='Use this info to name your cluster to make it easier to be identified')
def main(domain, token, clustervmtype, clusterid=None, clustername=None, terminate=False, permanent=False):
    global DOMAIN
    global TOKEN
    global CLUSTER_NAME
    global ClUSTER_ID
    global CLUSTER_VMTYPE
    DOMAIN = domain
    TOKEN = str.encode(token)
    CLUSTER_NAME = clustername
    ClUSTER_ID = clusterid

    if(clustervmtype is not None):
        CLUSTER_VMTYPE = clustervmtype

    if terminate is False:
        create_cluster()
        start_cluster()
        install_libraries()
        check_libraries()
        print("Cluster: %s is ready" % (CLUSTER_ID))
    else:
        if clusterid is not None:
            if permanent is False:
                terminate_cluster()
                print("Cluster: %s is terminated" % (CLUSTER_ID))
            else:
                permanent_terminate_cluster()
                print("Cluster: %s is permanently deleted" % (CLUSTER_ID))
        else:
            print("[Error]:Clusterid is required")
            sys.exit(1)

    sys.exit(CLUSTER_ID)


if __name__ == '__main__':
    main()
