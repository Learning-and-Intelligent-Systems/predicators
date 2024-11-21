import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# Set font aesthetics
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 16

# Define the data for each environment
data_combo_burger = {
    'EXPERIMENT_ID': [
        'VLM feat. pred', 'Ours', 'No feat.', 'No invent',
        'No subselect', 'No visual', 'VLM subselect', 
        'ViLa', 'ViLa fewshot', 'No noise tol.'
    ],
    'NUM_SOLVED': [0.00, 8.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 3.80],
    'NUM_SOLVED_STDDEV': [0.00, 1.17, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40]
}

data_fatter_burger = {
    'EXPERIMENT_ID': [
        'VLM feat. pred', 'Ours', 'No feat.', 'No invent',
        'No subselect', 'No visual', 'VLM subselect', 
        'ViLa', 'ViLa fewshot'
    ],
    'NUM_SOLVED': [0.00, 9.60, 1.20, 0.00, 0.00, 1.20, 3.00, 0.80, 3.80],
    'NUM_SOLVED_STDDEV': [0.00, 0.80, 2.40, 0.00, 0.00, 2.40, 1.41, 0.40, 0.40]
}

data_more_stacks = {
    'EXPERIMENT_ID': [
        'VLM feat. pred', 'Ours', 'No feat.', 'No invent',
        'No subselect', 'No visual', 'VLM subselect', 
        'ViLa', 'ViLa fewshot'
    ],
    'NUM_SOLVED': [0.00, 9.40, 0.00, 0.00, 0.00, 0.00, 3.60, 0.80, 3.80],
    'NUM_SOLVED_STDDEV': [0.00, 0.80, 0.00, 0.00, 0.00, 0.00, 2.24, 1.17, 0.40]
}

data_kitchen_boil_kettle = {
    'EXPERIMENT_ID': [
        'VLM feat. pred', 'Ours', 'No feat.', 'No invent',
        'No subselect', 'No visual', 'VLM subselect', 
        'ViLa', 'ViLa fewshot'
    ],
    'NUM_SOLVED': [0.00, 9.80, 9.80, 0.00, 0.00, 9.80, 1.00, 6.60, 10.00],
    'NUM_SOLVED_STDDEV': [0.00, 0.40, 0.40, 0.00, 0.00, 0.40, 2.00, 1.02, 0.00]
}

# Convert each dataset to a DataFrame
df_combo_burger = pd.DataFrame(data_combo_burger)
df_fatter_burger = pd.DataFrame(data_fatter_burger)
df_more_stacks = pd.DataFrame(data_more_stacks)
df_kitchen_boil_kettle = pd.DataFrame(data_kitchen_boil_kettle)

# Reorder the 'EXPERIMENT_ID' column to match 'custom_order'
custom_order = [
    'Ours', 'VLM subselect', 'No subselect', 'No feat.', 'No visual', 'No invent',
    'VLM feat. pred', 'ViLa', 'ViLa fewshot'
]

# Apply Categorical ordering before any transformations
for df in [df_combo_burger, df_fatter_burger, df_more_stacks, df_kitchen_boil_kettle]:
    df['EXPERIMENT_ID'] = pd.Categorical(df['EXPERIMENT_ID'], categories=custom_order, ordered=True)
    df.sort_values('EXPERIMENT_ID', inplace=True)

# Convert 'NUM_SOLVED' to percentages and calculate standard error
for df in [df_combo_burger, df_fatter_burger, df_more_stacks, df_kitchen_boil_kettle]:
    df['NUM_SOLVED'] = df['NUM_SOLVED'] * 10
    df['NUM_SOLVED_SE'] = df['NUM_SOLVED_STDDEV'] / np.sqrt(5) * 10

# Initialize subplots
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

# Assign a larger color palette for the bars, so that each bar has a unique color
unique_palette = sns.color_palette("pastel", n_colors=len(df_combo_burger))

# Plot in the new order: 'Boil Kettle', 'More Stacks', 'Bigger Burger', then 'Combo Burger'
environments = [df_kitchen_boil_kettle, df_more_stacks, df_fatter_burger, df_combo_burger]
titles = ["Kitchen Boil Kettle", "More Burger Stacks", "Bigger Burger", "Combo Burger"]

for i, (df, title) in enumerate(zip(environments, titles)):
    sns.barplot(
        data=df, y='EXPERIMENT_ID', x='NUM_SOLVED', ax=axes[i], palette=unique_palette, capsize=0.1
    )
    axes[i].errorbar(
        df['NUM_SOLVED'], df['EXPERIMENT_ID'],
        xerr=df['NUM_SOLVED_SE'], fmt='none', c='black', capsize=5, capthick=1
    )
    axes[i].set_title(title, fontsize=20)  # Increase title font size
    axes[i].set_xlabel('')  # Clear individual x-labels
    axes[i].set_ylabel('', fontsize=16)  # Increase y-label font size
    axes[i].tick_params(axis='both', labelsize=14)  # Increase tick label size
    axes[i].grid(True, linestyle='--', alpha=0.6)  # Add gridlines for clarity

# Set shared x-label
fig.text(0.5, 0.01, '% Evaluation Tasks Solved', ha='center', fontsize=18)

# Adjust layout with tighter spacing
plt.tight_layout(rect=[0.02, 0.05, 1, 1])
plt.show()