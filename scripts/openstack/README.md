# Instructions for Using Openstack

1. Request an account through the LIS group. See [here](https://tig.csail.mit.edu/shared-computing/open-stack/quick-start/).
2. Follow the [instructions for creating and uploading a private key](https://tig.csail.mit.edu/shared-computing/open-stack/openstack-ssh-key/).
3. Launch instances using the `predicators` image and your private key. Make sure to enable `allow_incoming_ssh` in the security settings when launching the instance so that the machine is actually accessible. Make sure you launch enough instances so that each (environment, approach, seed) can have one instance.
    1. You might need to create a new image. You can do this by launching an instance, ssh'ing it, installing everything you need, and then using the `Create Snapshot` Action on the instances dashboard. To ssh a machine, find its IP on openstack, and do `ssh ubuntu@<ip>` (be sure you're on the MIT network)!
4. Create a file (e.g. `machines.txt`) that lists your instance IP addresses, one per line. Do not include anything on the line that isn't the IP address. 
5. Create an experiment yaml file (see `scripts/openstack/configs/example_basic.yaml` for an example).
6. Run `python scripts/openstack/launch.py --config <config file> --machines <machines file> --sshkey <private key file>` to launch your experiments.
7. Wait for your experiments to complete.
8. Run `python scripts/openstack/download.py --dir <download dir> --machines <machines file> --sshkey <private key file>` to download the results.
