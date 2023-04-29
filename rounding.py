import numpy as np
from simulation import MyData
import copy
import math
from typing import Tuple

#|%%--%%| <JxLXF4kcb2|KVGcoAg890>

class Grids:

    def __init__(self, input_data: MyData, grid_width: float):
        self.grid_width = grid_width

        # Get the minimum and maximum of labels
        y_labels = input_data.y_labels
        self.min_label = np.floor(y_labels.min())
        self.max_label = np.ceil(y_labels.max())
        self.grids = self._generate_grids(grid_width)

    def _generate_grids(self, grid_width: float) -> np.ndarray:

        # Round the boundaries and then form the grids
        # Keep the boundaries fixed, not depending on the labels
        # scale = round(1 / grid_width) # Say 1/0.01 = 100
        min_rounded_label = np.floor(self.min_label)
        max_rounded_label = np.ceil(self.max_label) + grid_width * 3
        # grids = np.arange(min_rounded_label, max_rounded_label, grid_width)
        # num_grids = round((max_rounded_label - min_rounded_label) / grid_width)
        grid_left = np.floor(min_rounded_label / grid_width) * grid_width
        grid_right = np.ceil(max_rounded_label / grid_width) * grid_width
        # grids = np.linspace(min_rounded_label, max_rounded_label, num_grids + 1)
        grids = np.arange(grid_left, grid_right + 0.1, grid_width)
        return grids

    def update_grids(self, input_data: MyData):
        y_labels = input_data.y_labels
        min_label = y_labels.min()
        max_label = y_labels.max()
        if min_label < self.grids[0] or self.grids[-1] < max_label:
            self.min_label = np.minimum(np.floor(min_label), self.min_label)
            self.max_label = np.maximum(np.ceil(max_label), self.max_label)
            self.grids = self._generate_grids(self.grid_width)

    def round_labels(self, input_data: MyData) -> MyData:
        rounded_data = copy.deepcopy(input_data)
        labels = input_data.y_labels
        # Rounding label by label
        for i_label, this_label in enumerate(labels):
            # i_grid = np.argmin(np.abs(self.grids - this_label))
            i_grid = round((this_label - self.min_label) / self.grid_width)
            rounded_data.y_labels[i_label] = self.grids[i_grid]
        return rounded_data

    def round_labels_both(self, input_data: MyData) -> \
            Tuple[MyData, MyData, MyData, np.ndarray, np.ndarray]:
        rounded_data_nord = copy.deepcopy(input_data)
        rounded_data_rd = copy.deepcopy(input_data)
        rounded_data_i = copy.deepcopy(input_data)
        labels = input_data.y_labels
        uniform_samples = np.random.rand(len(self.grids) - 1)
        uniform_samples_i = np.random.rand(len(input_data.y_labels))
        # uniform_samples = np.random.rand(len(input_data.y_labels))
        # Rounding label by label
        for i_label, this_label in enumerate(labels):
            # diff = self.grids - this_label
            # i_grid_right = np.where(diff > 0)[0][0]
            i_grid_left = math.floor((this_label - self.grids[0]) / self.grid_width)
            portion_left = (this_label - self.grids[i_grid_left]) / self.grid_width

            # Test
            if i_grid_left >= len(self.grids):
                print(self.grids)
                print(this_label)


            # Rounding - deterministic
            rounded_data_nord.y_labels[i_label] = self.grids[i_grid_left] if portion_left < 0.5 else self.grids[i_grid_left + 1]

            
            # Rounding - randomness with grids
            # round_random = np.random.binomial(n=1, p=p_right)
            portion_left_random = uniform_samples[i_grid_left]
            round_down_random = int(portion_left < portion_left_random)
            rounded_data_rd.y_labels[i_label] = self.grids[i_grid_left] if round_down_random == 1 else self.grids[i_grid_left + 1]

            # Rounding - randomness with data
            portion_left_random = uniform_samples_i[i_label]
            round_down_random = int(portion_left < portion_left_random)
            rounded_data_i.y_labels[i_label] = self.grids[i_grid_left] if round_down_random == 1 else self.grids[i_grid_left + 1]

        return rounded_data_nord, rounded_data_rd, rounded_data_i, uniform_samples, uniform_samples_i
        



#|%%--%%| <KVGcoAg890|TjuHuV7NJm>

# tmp = np.array([-1, 1, 2])
# np.where(tmp > 0)[0][0]


# training_data = sample_dataset(7, 7) 
# training_data.y_labels
# grids_test = Grids(training_data, 0.5)
# grids_test.round_labels(training_data).y_labels
# grids_test.round_labels_randomly(training_data).y_labels
