"""Discretizing the model for conformal prediction"""

def clear_memo_cache(func):
    func.cache_clear()


# Change the directory
import os
os.chdir("/Users/lingui/Desktop/Conformal")

# Import basic modules
import numpy as np
from importlib import reload
import math
import copy
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

import simulation # Simulate the data
import models # Import the model
import rounding # Round the data
# Get the conformal prediction sets
# import predsets

reload(simulation)
reload(models)
reload(rounding)


Grids = rounding.Grids
sample_dataset = simulation.sample_dataset
OrdinaryLinearSquares = models.OrdinaryLinearSquares

#|%%--%%| <OSJlShRlQl|kWdD0KfbVk>

import argparse
parser = argparse.ArgumentParser(description='Description of your script')
# Add arguments
parser.add_argument('-e', '--expe', type=int,
                    help='Specify the number of experiments')
parser.add_argument('-t', '--train', type=int,
                    help='Specify the number of sample size')
parser.add_argument('-s', '--sd', type=float,
                    help='Specify the sd of noise')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
n_expes = 1 if args.expe is None else args.expe
n_train = 100 if args.train is None else args.train
noise_sigma = 0.1 if args.sd is None else args.sd

p_features = 2
grid_width_all=np.array([10, 8, 4, 3, 2.5, 2, 1, 0.5, 0.2, 0.1])
alpha = 0.9
model_class = OrdinaryLinearSquares
method_features = 1
method_labels = 2


#|%%--%%| <kWdD0KfbVk|5Dm2d2w3Ik>

def conduct_experiments(
        n_expes: int, 
        n_train: int, 
        p_features: int, 
        grid_width_all: np.ndarray, 
        alpha: float,
        model_class,
        method_features: int,
        method_labels: int,
        noise_sigma: float
        ):

    

    def get_interval_intersection_with_length(interval_one, interval_two):
        a, b = interval_one
        c, d = interval_two

        if b < c or d < a:
            return (None, None, 0)
        interval_left = max(a, c)
        interval_right = min(b, d)
        interval_length = interval_right - interval_left

        return (interval_left, interval_right, interval_length)


    def compute_quantile(this_model, alpha, n_train):
        y_predict = this_model.get_predicted_labels()
        residuals = abs(y_predict - this_model.training_data.y_labels)
        y_test_predicted = y_predict[-1]
        residual_test = residuals[-1]
        index_quantile = math.floor(alpha * n_train)
        this_quantile = np.sort(residuals)[index_quantile - 1]
        return this_quantile, y_test_predicted, residual_test

    def compute_quantile_two(training_data, this_model, alpha, n_train):
        y_predict = this_model.get_predicted_labels()
        residuals = abs(training_data.y_labels[:-1] - y_predict[:-1])
        y_test_predicted = y_predict[-1]
        residual_test = abs(training_data.y_labels[-1] - y_predict[-1])
        index_quantile = math.floor(alpha * n_train)
        this_quantile = np.sort(residuals)[index_quantile - 1]
        return this_quantile, y_test_predicted, residual_test


    def get_prediction_set_with_length(training_data_original, training_data, my_grids, *args):
        # Initialize the intervals
        n_grids = len(my_grids.grids)
        interval_length_sum = 0

        prediction_set_left = 1e5
        prediction_set_right = -1e5

        if len(args) != 0:
            if args[0] == "randomness":
                uniform_samples = args[1]
            elif args[0] == "randomness with data":
                this_portion = (args[1])[-1]

        # this_portion = uniform_samples_i[-1]
        for i_grid in range(n_grids):
            # Take the interval and find the margins
            grid_point = my_grids.grids[i_grid]
            if len(args) == 0:
                margin_left = grid_point - my_grids.grid_width / 2 if i_grid > 0 else grid_point
                margin_right = grid_point + my_grids.grid_width / 2 if i_grid < (n_grids - 1) else grid_point

            elif args[0] == "randomness":
                this_grid_width_left = (1 - uniform_samples[i_grid - 1]) * my_grids.grid_width if i_grid > 0 else 0
                this_grid_width_right = uniform_samples[i_grid] * my_grids.grid_width if i_grid < (n_grids - 1) else 0
                # print(this_grid_width_left, this_grid_width_right, my_grids.grid_width)
                margin_left = grid_point - this_grid_width_left
                margin_right = grid_point + this_grid_width_right

            elif args[0] == "randomness with data":
                this_grid_width_left = (1 - this_portion) * my_grids.grid_width if i_grid > 0 else 0
                this_grid_width_right = this_portion * my_grids.grid_width if i_grid < (n_grids - 1) else 0
                # print(this_grid_width_left, this_grid_width_right, my_grids.grid_width)
                margin_left = grid_point - this_grid_width_left
                margin_right = grid_point + this_grid_width_right

            else:
                print(args)
                raise ValueError("Error")
            # print(margin_left, grid_point, margin_right)

            # Turn Y_{n+1} to y and fit the model
            training_data_y = copy.deepcopy(training_data)
            training_data_y.y_labels[-1] = grid_point
            model_y = model_class(training_data_y)

            # Compute the quantile of the residuals with the predicted test label
            this_quantile, y_test_predicted, _ = \
                    compute_quantile_two(training_data_original, 
                                         model_y, alpha, n_train)

            # Determine the subset which is in the prediction set 
            interval_left, interval_right, interval_length = \
                    get_interval_intersection_with_length( 
                            (margin_left, margin_right), 
                            (y_test_predicted - this_quantile, 
                                y_test_predicted + this_quantile))

            if interval_left is not None and interval_left < prediction_set_left:
                prediction_set_left = interval_left
            if interval_right is not None and interval_right > prediction_set_right:
                prediction_set_right = interval_right


            # if len(args) == 0 and interval_left is not None:
                # np.set_printoptions(precision=4, suppress=True)
                # print(np.array([interval_left, interval_right, interval_length, margin_left, margin_right, y_test_predicted - this_quantile, y_test_predicted + this_quantile]))
                # last_interval_right = interval_right

            # Add the cumulative length of the prediction set0.9630.963
            # this_interval = [interval_left, interval_right]
            interval_length_sum += interval_length
        prediction_set = np.array([prediction_set_left, prediction_set_right])

        return prediction_set, interval_length_sum
    def obtain_prediction_set(training_data, model_round, my_grids, n_train, alpha, *args):

        training_data_rounded = model_round.training_data
            # cover_round[i_expe] = get_cover_results(model_round, alpha, n_train)
        quantile_round, _, residual_test_round = \
            compute_quantile_two(training_data, model_round, n_train, alpha)
        is_covered = 1 if residual_test_round <= quantile_round else 0
        # Compute the length of the prediction set
        prediction_set_round, this_prediction_set_width = \
            get_prediction_set_with_length(training_data, 
                                           training_data_rounded, my_grids, *args)

        # this_prediction_set_width = prediction_set_round[1] - prediction_set_round[0]
        return is_covered, prediction_set_round, this_prediction_set_width

    # def get_cover_results(this_model, alpha, n_train):
        # y_predict = this_model.get_predicted_labels()
        # residuals = abs(y_predict - this_model.training_data.y_labels)
        # test_order = np.argsort(residuals)[-1] + 1
        # is_cover = 1 if test_order <= alpha * n_train else 0
        # return is_cover 



    def run_one_experiment(a):

        # Initialize differences of estimated coefficients - array
        num_grids = len(grid_width_all)
        # diff_all = np.full((num_grids, n_expes), np.nan)
        # cover = np.full((num_grids, n_expes), np.nan)
        # cover_round = np.full((num_grids, n_expes), np.nan)
        # cover_round_rd = np.full((num_grids, n_expes), np.nan)
        # cover_round_i = np.full((num_grids, n_expes), np.nan)
        # prediction_set_width_full = np.full((num_grids, n_expes), np.nan)
        # prediction_set_width = np.full((num_grids, n_expes), np.nan)
        # prediction_set_width_rd = np.full((num_grids, n_expes), np.nan)
        # prediction_set_width_i = np.full((num_grids, n_expes), np.nan)
        diff_all = np.full((num_grids, 1), np.nan)
        cover = np.full((num_grids, 1), np.nan)
        cover_round = np.full((num_grids, 1), np.nan)
        cover_round_rd = np.full((num_grids, 1), np.nan)
        cover_round_i = np.full((num_grids, 1), np.nan)
        prediction_set_width_full = np.full((num_grids, 1), np.nan)
        prediction_set_width = np.full((num_grids, 1), np.nan)
        prediction_set_width_rd = np.full((num_grids, 1), np.nan)
        prediction_set_width_i = np.full((num_grids, 1), np.nan)

        # Generate samples
        training_data = sample_dataset(n_train, p_features, 
                                       method_features, method_labels,
                                       noise_sigma) 
        # test_data = sample_dataset(n_test, p_features)
        grid_list = [Grids(training_data, grid_width) 
            for grid_width in grid_width_all]
        # if i_expe == 0:
            # grid_list = [Grids(training_data, grid_width) 
                # for grid_width in grid_width_all]
        i_expe = 0
        for index_grid, grid_width in enumerate(grid_width_all):
            # grid_list[index_grid].update_grids(training_data)
            my_grids = grid_list[index_grid]
            """ --------------------------- Full conformal --------------------- """
            if index_grid == 0:
                model_full = model_class(training_data)
                # cover[i_expe] = get_cover_results(model_full, alpha, n_train)
                quantile_full, _, residual_test_full = \
                            compute_quantile(model_full, n_train, alpha)
                cover[index_grid, i_expe] = 1 if residual_test_full <= quantile_full else 0
                # Get the full conformal prediction set with a closed form

                # prediction_set_full, this_prediction_set_width_full = \
                    # model_full.get_full_conformal_set(alpha)
                # prediction_set_width_full[index_grid, i_expe] = this_prediction_set_width_full
            else:
                cover[index_grid, i_expe] = cover[0, i_expe]
                # prediction_set_width_full[index_grid, i_expe] = prediction_set_width_full[0, i_expe]

   
            """ ------- Round the data without and with randomness -------- """
            # TBD: may be two uniform_samples for two methods: without and with
            # randomness
            training_data_rounded, training_data_rounded_rd, training_data_rounded_i,\
                    uniform_samples, uniform_samples_i = \
                    my_grids.round_labels_both(training_data)
            # print(training_data_rounded.y_labels, training_data_rounded_rd.y_labels, training_data_rounded_i.y_labels)
            

            """ ---------------- no randomness ---------------------------- """
            model_round = model_class(training_data_rounded)
            cover_round[index_grid, i_expe], prediction_set_round, \
                this_prediction_set_width \
                        = obtain_prediction_set(training_data, model_round, 
                                                my_grids, n_train, alpha)
            prediction_set_width[index_grid, i_expe] = this_prediction_set_width
            # # cover_round[i_expe] = get_cover_results(model_round, alpha, n_train)
            # quantile_round, _, residual_test_round = \
                        # compute_quantile_two(training_data, model_round, n_train, alpha)
            # cover_round[index_grid, i_expe] = 1 if residual_test_round <= quantile_round else 0
            # # Compute the length of the prediction set
            # prediction_set_round, this_prediction_set_width = \
                # get_prediction_set_with_length(training_data, 
                                               # training_data_rounded, my_grids)
            # this_prediction_set_width = prediction_set_round[1] - prediction_set_round[0]
            # prediction_set_width[index_grid, i_expe] = this_prediction_set_width



            """ ------------- with randomness - grids --------------------- """
            model_round_rd = model_class(training_data_rounded_rd)
            cover_round_rd[index_grid, i_expe], prediction_set_round_i, \
                this_prediction_set_width_rd \
                        = obtain_prediction_set(training_data, model_round_rd, 
                                                my_grids, n_train, alpha, 
                                                "randomness", 
                                                uniform_samples)
            prediction_set_width_rd[index_grid, i_expe] = this_prediction_set_width_rd
            # # cover_round_rd[i_expe] = get_cover_results(model_round_rd, alpha, n_train)
            # quantile_round_rd, _, residual_test_round_rd = \
                        # compute_quantile_two(training_data, model_round_rd, n_train, alpha)
            # cover_round_rd[index_grid, i_expe] = 1 if residual_test_round_rd <= quantile_round_rd else 0
            # # Compute the length of the prediction set
            # prediction_set_round_rd, this_prediction_set_width_rd = \
                # get_prediction_set_with_length(training_data,
                                               # training_data_rounded_rd, 
                                               # my_grids, "randomness")

            # this_prediction_set_width_rd = prediction_set_round_rd[1] - prediction_set_round_rd[0]
            # prediction_set_width_rd[index_grid, i_expe] = this_prediction_set_width_rd
            """ ------------- with randomness - data --------------------- """
            model_round_i = model_class(training_data_rounded_i)
            cover_round_i[index_grid, i_expe], prediction_set_round_i, \
                this_prediction_set_width_i \
                        = obtain_prediction_set(training_data, model_round_i, 
                                                my_grids, n_train, alpha, 
                                                "randomness with data", 
                                                uniform_samples_i)
            prediction_set_width_i[index_grid, i_expe] = this_prediction_set_width_i

            if this_prediction_set_width < this_prediction_set_width_rd: 
                np.set_printoptions(precision=4, suppress=True)
                print(a, ":", grid_width, "---", 
                      np.array([this_prediction_set_width, this_prediction_set_width_rd]),
                      "<")
            else:
                np.set_printoptions(precision=4, suppress=True)
                print(a, ":", grid_width, "---", 
                      np.array([this_prediction_set_width, this_prediction_set_width_rd]),
                      ">")
            """ Compare """
            # Compare coefficients
            coef_true = training_data.coefficients
            coef_round = model_round.estimated_coefficients
            coef_round_rd = model_round_rd.estimated_coefficients
            # Record the difference of the norm
            diff_all[index_grid, i_expe] = (np.linalg.norm(coef_true - coef_round) 
                    - np.linalg.norm(coef_true - coef_round_rd))
        grid_num = [len(my_grids.grids) for my_grids in grid_list]

        # results = (prediction_set_full, prediction_set_round, prediction_set_round_i,
                   # diff_all, cover, cover_round, cover_round_rd, cover_round_i,
                   # prediction_set_width_full, prediction_set_width, 
                   # prediction_set_width_rd, prediction_set_width_i,
                   # grid_num)
        results = (cover, cover_round, cover_round_rd, cover_round_i,
                   prediction_set_width_full, prediction_set_width, 
                   prediction_set_width_rd, prediction_set_width_i,
                   grid_num)

        return results


    # for i_expe in range(n_expes):
        # run_one_experiment()

    # The number of threads to run in parallel
    num_threads = 8
    
    # Create a thread pool executor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit the calculate_square function for each input
        # This will distribute the work across the threads in the pool
        futures = [executor.submit(run_one_experiment, a) for a in range(n_expes)]
    
        # Retrieve the results as they become available
        results_all = [future.result() for future in futures]

    return results_all


import cProfile
pr = cProfile.Profile()
pr.enable()



# prediction_set_full, prediction_set_round, prediction_set_i, \
        # diff_all, cover, cover_round, cover_round_rd, cover_round_i, \
        # prediction_set_width_full, prediction_set_width,\
        # prediction_set_width_round, prediction_set_width_i, \
        # grid_num  \
            # = conduct_experiments(n_expes=n_expes, n_train=n_train, 
                                  # p_features=p_features, 
                                  # grid_width_all=grid_width_all, 
                                  # alpha=alpha,
                                  # model_class=model_class,
                                  # method_features=method_features,
                                  # method_labels=method_labels)
results_all = conduct_experiments(n_expes=n_expes, n_train=n_train, 
                                  p_features=p_features, 
                                  grid_width_all=grid_width_all, 
                                  alpha=alpha,
                                  model_class=model_class,
                                  method_features=method_features,
                                  method_labels=method_labels,
                                  noise_sigma=noise_sigma)
pr.disable()
pr.print_stats(sort="time")






#|%%--%%| <5Dm2d2w3Ik|BSrAn9b5lh>

num_grids = len(grid_width_all)
diff_all = np.full((num_grids, n_expes), np.nan)
cover = np.full((num_grids, n_expes), np.nan)
cover_round = np.full((num_grids, n_expes), np.nan)
cover_round_rd = np.full((num_grids, n_expes), np.nan)
cover_round_i = np.full((num_grids, n_expes), np.nan)
# prediction_set_width_full = np.full((num_grids, n_expes), np.nan)
prediction_set_width_round = np.full((num_grids, n_expes), np.nan)
prediction_set_width_rd = np.full((num_grids, n_expes), np.nan)
prediction_set_width_i = np.full((num_grids, n_expes), np.nan)
grid_num_all = np.full((num_grids, n_expes), np.nan)



for i_expe in range(n_expes):
    for i_method in range(4):
        cover[:, i_expe] = results_all[i_expe][0].reshape(num_grids)
        cover_round[:, i_expe] = results_all[i_expe][1].reshape(num_grids)
        cover_round_rd[:, i_expe] = results_all[i_expe][2].reshape(num_grids)
        cover_round_i[:, i_expe] = results_all[i_expe][3].reshape(num_grids)
        # prediction_set_width_full[:, i_expe] = results_all[i_expe][4].reshape(num_grids)
        prediction_set_width_round[:, i_expe] = results_all[i_expe][5].reshape(num_grids)
        prediction_set_width_rd[:, i_expe] = results_all[i_expe][6].reshape(num_grids)
        prediction_set_width_i[:, i_expe] = results_all[i_expe][7].reshape(num_grids)
        grid_num_all[:, i_expe] = results_all[i_expe][-1]

grid_num = np.max(grid_num_all, axis=1)



#|%%--%%| <BSrAn9b5lh|7Pw4M90WaI>

# np.set_printoptions(precision=4, suppress=True)
# print("Averaged results:")
# print("Full:", prediction_set_width_full.mean(axis=1), prediction_set_width_full.var(axis=1))
# prediction_set_width_round.mean(axis=1)
# print("Rounded:", prediction_set_width_round.mean(axis=1), prediction_set_width_round.var(axis=1))
# print("Debiased-d:", prediction_set_width_round.mean(axis=1), prediction_set_width_round.var(axis=1))
# print("Debiased-i:", prediction_set_width_i.mean(axis=1), prediction_set_width_i.var(axis=1))
# print("Coverage")
# print(cover.mean(axis=1), cover_round.mean(axis=1), cover_round_rd.mean(axis=1), cover_round_i.mean(axis=1))
# print("Estimates of coefficients")
# print(diff_all.mean(axis=1), diff_all.var(axis=1))
# print("Grid numbers:", grid_num)


#|%%--%%| <7Pw4M90WaI|8QYbrsQ0oo>

categories = np.array(["Full", "Round", "with grids", "with data"], dtype=object).reshape(-1, 1)

tmp = np.vstack((prediction_set_width_round.mean(axis=1), prediction_set_width_round.mean(axis=1), prediction_set_width_i.mean(axis=1)))
results_width = np.hstack((categories[1:], np.round(tmp, 3))) 
# print(results_width)

tmp = np.vstack((cover.mean(axis=1), cover_round.mean(axis=1), cover_round_rd.mean(axis=1), cover_round_i.mean(axis=1)))
results_cover = np.hstack((categories, np.round(tmp, 3))) 
# print(results_cover)

oracle_width = stats.norm.ppf(1 - (1 - alpha) / 2, scale=noise_sigma) * 2
# print(oracle_width)




from csv import writer

grid_width_info = np.concatenate((np.array(['Width']), grid_width_all))
grid_num_info = np.concatenate((np.array(['#Grids']), grid_num))

filename = 'output.csv'

# Open our existing CSV file in append mode
# Create a file object for this file
with open(filename, 'a') as f_object:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(["-----"] * 7)
    writer_object.writerow(["method_features", method_features, "method_labels", method_labels])
    writer_object.writerow(["N", n_expes, "n", n_train, "Noise Sd", noise_sigma])
    writer_object.writerow(grid_width_info)
    writer_object.writerow(grid_num_info)
    # Write the data to the CSV file row by row
    for row in results_width:
        writer_object.writerow(row)

    writer_object.writerow(["Oracle", np.round(oracle_width, 3)])
    for row in results_cover:
        writer_object.writerow(row)
    # Close the file object
    f_object.close()

print(f"Data exported to {filename} successfully.")







