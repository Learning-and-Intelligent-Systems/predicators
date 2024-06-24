import logging

import matplotlib.pyplot as plt
import numpy as np

from predicators.settings import CFG
import pybullet as p

def visualize(sampler, x=None, rng=None):
    x = np.array([1., 1.35345304, 0.6483832, 0.7, 1., 0.80552434,
         0.93068886, 0.78291946, 1.35000002, 0.75, 0.69999999, 0.]) if x is None else x
    rng = np.random.default_rng(CFG.seed) if rng is None else rng
    samples = 1000
    x_plot = []
    y_plot = []
    logging.info(f"Plotting:")
    for sample in range(samples):
        sample_output = sampler.predict_sample(x, rng)
        #logging.info(f'SAMPLE OUTPUT: {sample_output}')
        x_plot.append(sample_output[0])

        if len(sample_output) == 1:
            y_plot.append(0)
        else:
            y_plot.append(sample_output[1])

    # Define the data for the straight line
    x_values = x_plot  # X-axis values
    y_values = y_plot  # Corresponding Y-axis values

    # Plot the straight line
    plt.scatter(x_values, y_values, label=f'{CFG.sampler_learning_regressor_model}')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Plot of {CFG.sampler_learning_regressor_model} dist')

    # Add legend
    plt.legend()

    # Display the plot
    plt.savefig(f"{CFG.env}_{CFG.sampler_learning_regressor_model}_plot.png", format="png")
    plt.close()
