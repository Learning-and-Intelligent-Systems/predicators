# Spot Utils

## Perception

> Last updated: 07/17/2023

We are currently using Detic + Segmentation Anything to provide text-conditioned bounding box + detection-conditioned segmentation mask.

We currently use REST interface from the `add-scores-to-output` branch of the [DETIC SAM BDAI repo](https://github.com/bdaiinstitute/detic-sam/tree/add-scores-to-output).

Note that BDAI is currently transitioning to using `torchserve`, so we likely need to update the client code a bit. 

The pipeline is as follows:
- Set up the repo
  - See the instructions in the above-linked repo. Note that you need to use the branch listed above.
- Run the server by running `server.py` from the above-linked repo
  - You only need to run the server once
  - It is recommended to run on a local computer (faster connection) with CUDA GPU (faster inference)
- Connect to server from local
  - Use SSH "local port forward"
  - `ssh -L 5550:localhost:5550 10.17.1.102`
- Request from your local computer
  - You can see perception_utils.py, or the `client.py` function in the BDAI repo.