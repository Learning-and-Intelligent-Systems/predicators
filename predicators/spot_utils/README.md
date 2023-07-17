# Spot Utils

## Perception

> Last updated: 07/17/2023

We are currently using Detic + Segmentation Anything to provide text-conditioned bounding box + detection-conditioned segmentation mask.

We currently use REST interface from BDAI repo `https://github.com/bdaiinstitute/detic-sam`.

Note that BDAI is currently transitioning to using `torchserve`, so we likely need to update the client code a bit. 

The pipeline is as follows:
- Set up the server
  - You only need to run the server once
  - See the instructions there
- Run the server by running `server.py` from the above-linked repo
  - It is recommended to run on a local computer (faster connection) with CUDA GPU (faster inference)
- Connect to server from local
  - Use SSH "local port forward"
  - `ssh -L 5550:localhost:5550 <IP-ADDRESS>`
- Request from your local computer
  - You can see perception_utils.py, or the `client.py` function in the BDAI repo.