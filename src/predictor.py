#!/usr/bin/python3 

from math import sqrt
import numpy as np
import sys
import re
import math
import random
import time
import utils
from config import config, read_config

########################################
# Prediction
########################################

class NoSuchDataLine(Exception):
    pass

def func(row_to_predict, data_learn, y_d, y_pow):
    pow_sign = 1.0
    if y_pow < 0:
        pow_sign = -1.0
    res = pow_sign * np.sum((np.sqrt(np.sum((row_to_predict - data_learn) ** 2, axis=1)) + y_d) ** y_pow)
    return res

def make_prediction(row_to_predict, data_learn, y_d, y_pow, ys_to_try):
    '''
    Returns a tuple
        ( y actual if it's given, None otherwise,
         y from ys_to_try where best local minimum is found or None if no such point is found
          value of func() at that point of None if y is None
          list of values of func() at all the rest of the points from ys_to_try,
          list of points where func() has a local min)
    '''

    func_values = [func(np.append(y, row_to_predict[1:]), data_learn, y_d, y_pow) for y in ys_to_try]
    local_mins = []
    for i in range(1, len(ys_to_try) - 1):
        if func_values[i - 1] >= func_values[i] and func_values[i + 1] >= func_values[i]:
            local_mins.append((ys_to_try[i], func_values[i]))
    if len(local_mins) > 0:
        best_local_min = min(local_mins, key=lambda a: a[1])
    else:
        best_local_min = (None, None)
    y_local_mins = list(map(lambda x: x[0], local_mins))
    return (row_to_predict[0], best_local_min[0], best_local_min[1], func_values, y_local_mins)

def generate_ys_to_try(y_min, y_max, y_step):
    res = []
    t = y_min
    while t <= y_max:
        res.append(t)
        t += y_step
    return res

def compute_column_stds(data):
    return np.std(data, axis=0)

def compute_column_mean_distances(data):
    D = np.zeros(data.shape)
    n = D.shape[0]
    for row_index in range(n):
        if row_index % 100 == 0:
            print("%.3f percent complete" % (row_index / n * 100.0))
        row = data[row_index, :]
        D += np.abs(data - row)
    return np.sum(D, axis=0) / (n * (n - 1))


########################################
# Main function
########################################

def compute_scalers(row_to_predict, data_learn, ys_to_try):
    def calc_func_with_y_to_try(y_to_try, row_to_predict):
        row = row_to_predict.copy()
        row[0] = y_to_try
        return func(row, data_learn, config.y_d, config.y_pow)

    row_to_predict_with_y_to_try = row_to_predict.copy()
    dim = data_learn.shape[1]

    # Sort the ys_to_try in order of descreasing value of the functional
    ys_to_try_sorted = sorted(ys_to_try, key=lambda y_to_try: calc_func_with_y_to_try(y_to_try, row_to_predict),
                              reverse=True)

    # Each column - shift, number of rows equals data_learn.shape[1], number of xs + 1
    best_shifts = np.zeros((dim, len(ys_to_try)))
    best_rows_directions = np.zeros((dim, len(ys_to_try)))
    best_func_deltas = np.zeros(len(ys_to_try))
    func_values = np.zeros(len(ys_to_try))
    best_directions_to_from = np.zeros(len(ys_to_try))
    rows_directions_num_float = float(config.directions_initial_num)

    # Matrix with config.directions_best_from_prev_step rows, each row is a point-direction
    best_rows_directions_from_prev_step = np.zeros((0, dim))

    y_to_try_prev = None
    for ys_to_try_index in range(len(ys_to_try_sorted)):
        y_to_try = ys_to_try_sorted[ys_to_try_index]
        if config.debug:
            print('Compute scalers: processing with y %d / %d' % (ys_to_try_index, len(ys_to_try_sorted)))

        rows_directions_num = min(data_learn.shape[0], math.floor(rows_directions_num_float))
        rows_directions_num_from_previous_step = best_rows_directions_from_prev_step.shape[0]
        # TODO: same rows-directions may appear in rows_directions matrix. Maybe need to add only new rows?
        rows_directions = data_learn[utils.random_indeces(data_learn.shape[0], rows_directions_num), :]

        if y_to_try_prev is not None and abs(y_to_try - y_to_try_prev) == 1:
            rows_directions = np.row_stack((rows_directions, best_rows_directions_from_prev_step))

        # Add the same points-directions, but with negated coordinates
        rows_directions = np.row_stack((rows_directions, -rows_directions))

        # if config.debug:
        #     print('Number of directions: ' + str(rows_directions_num) + ' + ' +
        #                                          str(rows_directions_num_from_previous_step) + ' = ' +
        #           str(rows_directions_num + rows_directions_num_from_previous_step))

        row_to_predict_with_y_to_try[0] = y_to_try
        row_to_predict_shifted, shifts = utils.shift_point_to_directions(row_to_predict_with_y_to_try, rows_directions)
        shifts /= config.unit_step

        func_value_initial = func(row_to_predict_with_y_to_try, data_learn, config.y_d, config.y_pow)
        func_values[ys_to_try_index] = func_value_initial
        func_values_shifted = [func(row_shifted, data_learn, config.y_d, config.y_pow) for row_shifted in
                        row_to_predict_shifted]
        best_shift_index = 0
        indeces_deltas = []
        # negative_deltas = []
        # positive_deltas = []
        for shift_index in range(len(func_values_shifted)):
            delta = func_value_initial - func_values_shifted[shift_index]
            indeces_deltas.append((shift_index, abs(delta)))
            # if delta < 0:
            #     negative_deltas.append(delta)
            # else:
            #     positive_deltas.append(delta)
            if abs(delta) > abs(func_values_shifted[best_shift_index] - func_value_initial):
                best_shift_index = shift_index
        # Best shift - the shift, which leads to the greatest change in value of the functional

        best_directions_to_from[ys_to_try_index] = 1 if best_shift_index < rows_directions.shape[0] // 2 else -1
        if func_values_shifted[best_shift_index] != func_value_initial:
            best_shift = shifts[best_shift_index, :].reshape((data_learn.shape[1]))
            best_rows_direction = rows_directions[best_shift_index, :]
            best_func_deltas[ys_to_try_index] = func_value_initial - func_values_shifted[best_shift_index]
        else:
            best_shift = np.zeros(dim)
            best_rows_direction = np.zeros(dim)
            best_func_deltas[ys_to_try_index] = 0.0
        best_shifts[:, ys_to_try_index] = np.abs(best_shift)
        best_rows_directions[:, ys_to_try_index] = best_rows_direction

        # Sort shifts from the highest functional delta to the lowest
        indeces_deltas = list(filter(lambda x: x[0] < rows_directions.shape[0] // 2, indeces_deltas))
        indeces_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
        rows_directions_num_for_the_next_iteration = min(rows_directions.shape[0] // 2,
                                                         config.directions_best_from_prev_step)
        best_rows_directions_from_prev_step = np.zeros((rows_directions_num_for_the_next_iteration, dim))
        for i in range(rows_directions_num_for_the_next_iteration):
            best_rows_directions_from_prev_step[i,:] = rows_directions[indeces_deltas[i][0],:]

        rows_directions_num_float += config.directions_increase
        y_to_try_prev = y_to_try

    best_func_deltas_row = best_func_deltas.reshape((1, len(ys_to_try)))
    if config.use_squares_of_deltas:
        weightedSumResults = np.sum(best_shifts * (best_func_deltas_row ** 2), axis=1)
        sum_func_deltas = np.sum(best_func_deltas_row ** 2)
        if sum_func_deltas > 0:
            weightedSumResults /= sum_func_deltas
    else:
        weightedSumResults = np.sum(best_shifts * np.abs(best_func_deltas_row), axis=1)
        sum_func_deltas = np.sum(np.abs(best_func_deltas_row))
        if sum_func_deltas > 0:
            weightedSumResults /= sum_func_deltas

    scalers = weightedSumResults.copy()
    # "Compress" scalers and normalize it on y's scaler
    for i in range(scalers.size):
        scalers[i] = (config.lowest_scaler_bound + (1.0 - config.lowest_scaler_bound) * scalers[i])

    if scalers[0] > 0:
        scalers /= scalers[0]

    return scalers, weightedSumResults, best_shifts, best_func_deltas, best_rows_directions, ys_to_try_sorted, \
           func_values, best_directions_to_from


def main():
    read_config(config.config_filename)
    all_data, all_data_ids_map = utils.read_data(config.input_filename, config.input_separator, config.columns_num)

    ############ BEGIN: build learn dataset
    if not config.ids_rows_learn:
        config.ids_rows_learn = all_data_ids_map.keys()
    if config.read_learn_data_from_file:
        data_learn, data_learn_id_map = utils.read_data(config.learn_data_filename, config.input_separator, config.columns_num)
        config.ids_rows_learn = data_learn_id_map.keys()
    else:
        data_learn = all_data[[all_data_ids_map[row_id] for row_id in config.ids_rows_learn], :]
    ############ END: build learn dataset

    ############  BEGIN: Compute normalizers & normalize data
    if config.debug:
        print('Start normalizing data...')

    column_vars = compute_column_stds(data_learn)
    if config.normalize_by_mean_distance:
        column_mean_distances = compute_column_mean_distances(data_learn)
        column_normalizers = column_mean_distances.copy()
    else:
        column_normalizers = column_vars.copy()

    for i in range(column_normalizers.shape[0]):
        if column_normalizers[i] == 0.0:
            column_normalizers[i] = 1.0

    if config.normalizing:
        utils.normalize_data(all_data, column_normalizers)
        utils.normalize_data(data_learn, column_normalizers)
    ############  END: Compute normalizers & normalize data

    ############  BEGIN: Filter learn dataset
    data_learn_reduced = data_learn
    if not config.read_learn_data_from_file and config.use_reduction:
        if config.debug:
            print('Started filtering data...')
        ids_rows_learn_reduced = utils.reduce_data_set(config.ids_rows_learn, config.use_reduction_size)
        data_learn_reduced = all_data[[all_data_ids_map[row_id] for row_id in ids_rows_learn_reduced], :]
        config.ids_rows_learn = ids_rows_learn_reduced

    if config.debug:
        print('Saving filtered learning data to %s' % config.reduced_rows_filename)
    utils.print_data_rows(config.ids_rows_learn, data_learn, separator=config.input_separator, file=config.reduced_rows_filename,
                    useCommaInNumbers=config.output_numbers_use_comma)
    ############  END: Filter learn dataset

    ys_to_try = np.arange(config.y_min, config.y_max, config.y_step)

    for y_index in config.columns_ys:
        # For each of ys specified in 'column_ys' option...
        # (usually it's just one y)

        if config.debug:
            print('Preparing data for y index %d' % (y_index + 1))

        ############ BEGIN: Prepare data array for y_index
        columns_xs = [i for i in range(config.columns_num) if i not in config.columns_ys]
        all_data_for_index = np.column_stack((all_data[:, y_index], all_data[:, columns_xs]))
        data_learn_for_index = np.column_stack((data_learn[:, y_index], data_learn[:, columns_xs]))
        data_learn_for_index_reduced = np.column_stack((data_learn_reduced[:, y_index], data_learn_reduced[:, columns_xs]))
        prescalers = [config.prescalers[y_index]] + [config.prescalers[i] for i in columns_xs]
        ############ END: Prepare data array for y_index

        ############ BEGIN: prescale data
        all_data_for_index_prescaled = all_data_for_index.copy()
        data_learn_for_index_prescaled_reduced = data_learn_for_index_reduced.copy()
        utils.scale_data(all_data_for_index_prescaled, prescalers)
        utils.scale_data(data_learn_for_index_prescaled_reduced, prescalers)
        ############ END: prescale data

        # Now 'data_learn_for_index' contains data rows, with each row in the following format:
        # y, x_1, ..., x_n

        if config.debug:
            print('Done')

        of = sys.stdout
        if config.output_filename:
            try:
                of = open(utils.generate_output_filename(config.output_filename, y_index + 1), 'w')
            except IOError:
                utils.die('Can\'t open output file \'', utils.generate_output_filename(config.output_filename,
                                                                                       y_index + 1), '\'')

        if config.debug:
            print('Predicting for y with index %d' % (y_index + 1))

        utils.print_header(ys_to_try, len(columns_xs), file=of, separator=config.output_separator, \
                                 useCommaInNumbers=config.output_numbers_use_comma,
                                 printWeightedSumFormulaResults=config.output_weighted_sum_formula_results,
                                 printFuncValues=config.output_func_values)

        ys_actual = []
        ys_predicted = []

        for row_id in config.rows_predict:
            # For each row specified in 'predict' option...

            row_to_predict = all_data_for_index_prescaled[all_data_ids_map[row_id]]

            t1 = time.time()
            ############  BEGIN: Compute scalers
            data_learn_for_index_prescaled_reduced_scaled = data_learn_for_index_prescaled_reduced
            if config.compute_scalers:
                if config.debug:
                    print('Computing scalers for row #' + str(row_id))
                scalers, weightSumResults, best_shifts, best_func_deltas, best_row_directions, ys_to_try_sorted, \
                func_values, directions = \
                    compute_scalers(
                    row_to_predict, data_learn_for_index, ys_to_try)

                if config.test_output:
                    utils.print_test_output(row_id, row_to_predict, ys_to_try_sorted, best_row_directions, best_shifts,
                                            best_func_deltas, func_values, directions,
                                            useCommaInNumbers=config.output_numbers_use_comma,
                                            separator=config.output_separator)

                scalers[scalers == 0.0] = 1.0
                data_learn_for_index_prescaled_reduced_scaled = data_learn_for_index_prescaled_reduced.copy()
                utils.scale_data(data_learn_for_index_prescaled_reduced_scaled, scalers)
                utils.scale_data(row_to_predict, scalers)
            else:
                scalers = np.array([1.0 for _ in range(len(columns_xs) + 1)])
                weightSumResults = np.array([None for _ in range(len(columns_xs) + 1)])
            ############  END: Compute scalers

            if config.debug:
                print('Predicting row #' + str(row_id))
            try:
                # Make prediction for the current row
                y_actual, y_best, func_best, func_values, y_local_minims = \
                    make_prediction(row_to_predict, data_learn_for_index_prescaled_reduced_scaled, config.y_d, config.y_pow, ys_to_try)
                t2 = time.time()
                if config.debug:
                    print('Elapsed %.6f ms' % (((t2 - t1)) * 1000))

                ys_actual.append(y_actual)
                ys_predicted.append(y_best)

                #### BEGIN: Denormalizing results
                y_actual_denormalized = y_actual / prescalers[0]
                y_actual_denormalized = y_actual_denormalized / scalers[0]
                if config.normalizing:
                    y_actual_denormalized *= column_normalizers[0]
                y_best_denormalized = y_best / prescalers[0]  if y_best is not None else None
                y_best_denormalized = y_best_denormalized / scalers[0] if y_best is not None else None
                if config.normalizing and y_best_denormalized is not None:
                    y_best_denormalized *= column_normalizers[0]
                y_local_minims_denormalized = list(map(lambda y: y / prescalers[0],
                                                       y_local_minims))
                y_local_minims_denormalized = list(map(lambda y: y / scalers[0],
                                                       y_local_minims_denormalized))
                if config.normalizing:
                    y_local_minims_denormalized = list(map(lambda x: x * column_normalizers[0],
                                                       y_local_minims_denormalized))
                #### END: Denormalizing results

                # Print prediction for the current row
                utils.print_prediction(row_id, y_actual, y_actual_denormalized, y_best, y_best_denormalized,
                                       scalers, weightSumResults, func_values, \
                                 y_local_minims, \
                                 y_local_minims_denormalized, \
                                 file=of, \
                                 printWeightedSumFormulaResults=config.output_weighted_sum_formula_results,
                                 printFuncValues=config.output_func_values,
                                 separator=config.output_separator, \
                                 useCommaInNumbers=config.output_numbers_use_comma)

            except NoSuchDataLine:
                print('Can\'t predict line #' + str(row_id)
                      + ': no such line in data!')
                exit(1)

        utils.print_common_statistics(column_mean_distances, column_vars, config.columns_num, columns_xs,
                                 config.normalize_by_mean_distance,
                                of, y_index, ys_actual, ys_predicted)

        if config.output_filename:
            of.close()

if __name__ == '__main__':
    main()
    input('Press Enter to quit...')
