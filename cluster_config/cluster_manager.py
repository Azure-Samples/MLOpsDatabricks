import sys
import os
import time
import argparse
from cluster import DatabricksCluster
from cluster import ClusterManagementException


def parse_arguments():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--cluster-name',
        help='Gives the created cluster a suffix to its name'
    )
    arg_parser.add_argument(
        '--terminate',
        action='store_true',
        help='Terminates cluster,requires --clusterid')
    arg_parser.add_argument(
        '--permanent',
        action='store_true',
        help='Permanently deletes cluster, works with --terminate')

    return arg_parser.parse_args()


def terminate_cluster(
    cluster: DatabricksCluster,
    databricks_access_token,
    libraries_config_path,
    permanent: bool
):
    if cluster.id is not None:
        try:
            cluster.terminate(
                databricks_access_token,
                libraries_config_path,
                permanent
            )
        except ClusterManagementException as e:
            print(str(e))
            sys.exit(1)

        if permanent is False:
            print("Cluster %s is terminated" % (cluster.id))
        else:
            print("Cluster %s is permanently deleted" % (cluster.id))
    else:
        print("[Error]:Clusterid is required")
        sys.exit(1)


def provision_cluster(
    cluster: DatabricksCluster,
    databricks_access_token,
    libraries_config_path,
    databricks_cluster_vmtype
):
    try:
        if cluster.id is None or not cluster.id:
            if databricks_cluster_vmtype is not None:
                cluster.create(
                    databricks_access_token,
                    databricks_cluster_vmtype
                )
            else:
                cluster.create(databricks_access_token)
            print("Requested to create the cluster...")
        else:
            cluster.start(databricks_access_token)
            print(
                "Requested to start the cluster with id %s" %
                (cluster.id)
            )

        while cluster.state == 'PENDING':
            print("Cluster %s is pending..." % (cluster.id))
            time.sleep(30)
            cluster.get_state(databricks_access_token)

        cluster.install_libraries(
            databricks_access_token,
            libraries_config_path
        )
        print("Installing libraries on %s..." % (cluster.id))

        libraries_status = cluster.check_libraries(databricks_access_token)
        while libraries_status == 'INSTALLING':
            time.sleep(30)
            print("Installing libraries on %s..." % (cluster.id))
            libraries_status = cluster.check_libraries(
                databricks_access_token
            )

        print(
            "Libraries installed and verified on cluster %s" %
            (cluster.id)
        )

        print("Cluster %s is ready" % (cluster.id))
    except ClusterManagementException as e:
        print(str(e))
        sys.exit(1)


def get_cluster_name():
    cluster_name_suffix_var = os.environ.get(
        "DATABRICKS_CLUSTER_NAME_SUFFIX",
        None
    )

    if cluster_name_suffix_var is None:
        return None

    if cluster_name_suffix_var.startswith('ENV_'):
        environment_variable_name = os.environ.get(
            "DATABRICKS_CLUSTER_NAME_SUFFIX"
        ).split('ENV_', 2)[1]

        databricks_cluster_name_suffix = os.environ.get(
            environment_variable_name,
            None
        )
    else:
        databricks_cluster_name_suffix = cluster_name_suffix_var

    return databricks_cluster_name_suffix


def main():
    args = parse_arguments()

    databricks_domain = os.environ.get("DATABRICKS_DOMAIN", None)
    databricks_access_token = os.environ.get(
        "DATABRICKS_ACCESS_TOKEN",
        None)
    databricks_cluster_vmtype = os.environ.get(
        "DATABRICKS_CLUSTER_VMTYPE",
        "Standard_D3_v2"
    )

    databricks_cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID", None)

    # If databricks_cluster_id is not None, but it's an empty string: its None
    if databricks_cluster_id is not None and not databricks_cluster_id:
        databricks_cluster_id = None

    databricks_cluster_name_suffix = get_cluster_name()

    cluster = DatabricksCluster(
        databricks_cluster_id,
        databricks_cluster_name_suffix,
        databricks_domain
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    libraries_config_path = os.path.join(script_dir, 'library.json')

    if args.terminate is False:
        provision_cluster(
            cluster,
            databricks_access_token,
            libraries_config_path,
            databricks_cluster_vmtype
        )
    else:
        terminate_cluster(
            cluster,
            databricks_access_token,
            libraries_config_path,
            args.permanent
        )

    sys.exit(cluster.id)


if __name__ == '__main__':
    main()
