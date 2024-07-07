"""Open-loop vision-language model (VLM) meta-controller approach.
The idea here is that the VLM is given a set of training trajectories
consisting of the initial state image as well as the low-level
object-oriented state, and then asked to output a full plan.
TODO: we probably want to ensure the model is prompted with entire
trajectories (so all intermediate states), and not just the
initial state and then the plan. we also probably want to not
have to do sampler learning somehow?

"""