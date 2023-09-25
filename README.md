# Sequential Latent Variable Agent (QSSM)

Implementation for learning a world model and a Q-value model through variational inference. The agent uses a simple MPC planner (RHE) to harness the approximate world model.

Support for:

- Gaussian RSSM
- Categorical RSSM


Other features
- Action is passed on explicitly for the reward and pcont decoder
- Learn final observation


## Buffers

full_trajectory_buffer: this buffer considers the additional flag i, to facilitate storing terminal observations.

