import os

input_size = 42 # 6-dimensional user CS, 8-dimensional agent CS, 2-dimensional previous rapport values
hidden_size = 8
output_size = 7 # 6-dimensional user CS (at next step), 1-dimensional rapport value
window = 2 # How many previous user and agent CSs and rapport values we take into account for prediction
leaky_slope = 0.1 # Hyperparameter for Leaky ReLU
parent_path = os.path.abspath('../')
data_path = parent_path + '/src/data/'