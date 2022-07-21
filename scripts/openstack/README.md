# Instructions for Using Openstack

1. Request an account through the LIS group. See [here](https://tig.csail.mit.edu/shared-computing/open-stack/quick-start/).
2. Follow the instructions for creating and uploading a private key.
3. Launch instances using the `predicators` image with your private key. Make sure you launch enough instances so that each (environment, approach, seed) can have one instance.
4. Create a file (e.g. `machines.txt`) that lists your instance IP addresses, one per line.
5. Create an experiment yaml file (see `scripts/openstack/configs/example.yaml` for an example).
6. Run `python scripts/openstack/launch.py --config <config file> --machines <machines file> --sshkey <private key file>` to launch your experiments.
7. Wait for your experiments to complete.
8. Run `python scripts/openstack/download.py --dir <download dir> --machines <machines file> --sshkey <private key file>` to download the results.
