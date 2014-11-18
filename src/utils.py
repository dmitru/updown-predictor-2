__author__ = 'dmitry'

from math import sqrt
import re
import sys
import numpy as np
from random import random
from random import randint
from random import shuffle
from config import config

########################################
# Utilities
########################################

def list_to_str(l, left = '[ ', right = ' ]', delim = ' ', noneReplacement='-', useCommaInNumbers=False):
    s = left
    i = 0
    n = len(l)
    for el in l:
        if el is None:
            s += noneReplacement
        elif type(el) is int or type(el) is str:
            s += str(el)
        else:
            el = float(el)
            s += '%.10f' % el
            if useCommaInNumbers:
                s = s.replace('.', ',')
        if i < n - 1:
            s += delim
        i += 1
    s += right
    return s

def die(error_msg):
    '''Terminate program with error message error_msg.'''
    print('Error:', error_msg + '.')
    exit(1)

def compute_dist(p1, p2):
    '''Returns the distance between two points'''
    return sqrt(sum(map(lambda x: (x[0] - x[1])**2,
                        zip(p1, p2))))

def np_compute_sqr_dist(x, y):
    return np.sum((x - y)**2)

def generate_output_filename(filename, index):
    without_extension = re.sub('\.\S+$', '', filename)
    extension = re.findall('\.(\S+)', filename)[0]
    return '%s_%d.%s' % (without_extension, index, extension)

def print_data_rows(row_ids, data_rows, separator=' ', useCommaInNumbers=False, file=None):
    f = sys.stdout
    if file:
        f = open(file, 'w')
    i = 0
    for row_id in row_ids:
        print(str(row_id), list_to_str(data_rows[i], left='', right='', delim=separator, \
                                       useCommaInNumbers=useCommaInNumbers), sep=separator, file=f)
        i += 1
    if file:
        f.close()

def print_header(ys_to_try, num_xs, file=None, separator=' ', useCommaInNumbers=False, printFuncValues=True,
                 printWeightedSumFormulaResults=True):
    print(list_to_str(['id', 'actual', 'actual_normalized', 'best_y', 'best_y_normalized'] + \
                      ['scaler_1'] + ['scaler_%d' % (i + 2) for i in range(num_xs)] + \
                      ((['weighted_sum_1'] + ['weighted_sum_%d' % (i + 2) for i in range(num_xs)]) if
                      printWeightedSumFormulaResults else []) + \
                      (list(ys_to_try) if printFuncValues else []),
                      left='',
                      right='',
                      delim=separator,
                      useCommaInNumbers=useCommaInNumbers),
          file=file)

def alternate_lists(a, b):
    res = []
    for i in range(min(len(a), len(b))):
        res.append(a[i])
        res.append(b[i])
    return res


########################################
# Input data
########################################

def read_data(fname, input_separator, columns_num, debug=True):
    ''' Returns data in np array
    each row contains id, x_i_1, ..., x_i_n
    and map from row_id to row_index in this array
    '''

    if debug:
        print('Start reading data...')

    try:
        df = open(fname)
    except IOError:
        die("Can't open data file '" + fname + "'")

    lines = []
    for l in df:
        l = l.strip()
        if not l:
            continue # Ignore empty lines
        else:
            lines.append(l)
    df.close()

    data = np.zeros((len(lines), columns_num))
    data_by_id = {}
    row_index = 0
    for l in lines:
        try:
            row = list(map(lambda el: el.strip(), l.split(input_separator)))
            if len(row) < 1 + columns_num:
                raise ValueError()
            row[0] = int(row[0])
            for i in range(1, columns_num + 1):
                row[i] = float(re.sub(',', '.', row[i]))

            data[row_index] = row[1 : columns_num + 1]
            data_by_id[int(row[0])] = row_index
        except ValueError:
                print('Ignored incorrect data line: "' + l + '".')
        row_index += 1
    if data.shape[0] == 0:
        die('No data to be processed')
    if debug:
        print()
        print(len(data), 'data lines in total.')

    if len(data_by_id) != len(data):
        die('In data some IDs (numbers in the first column) are not unique')

    return data, data_by_id

########################################
# Output
########################################

def print_prediction(row_id, y_actual, y_actual_denormalized, y_best, y_best_denormalized, \
                     scalers, weightedSumResults, func_values,
                     y_local_minims, y_local_minims_denormalized, \
                     file, printFuncValues=True, \
                     printWeightedSumFormulaResults=True, useCommaInNumbers=False, \
                     separator=' '):
    print(list_to_str( \
        [row_id, y_actual_denormalized, y_actual, y_best_denormalized, y_best] + \
        scalers.tolist() + (weightedSumResults.tolist() if printWeightedSumFormulaResults else []) + \
        (func_values if printFuncValues else []) + \
        alternate_lists(y_local_minims, y_local_minims_denormalized), \
        left='', right='', useCommaInNumbers=useCommaInNumbers, delim=separator), file=file)

def print_common_statistics(column_mean_distances, column_vars, columns_num, columns_xs, normalize_by_mean_distance, of,
                            y_index, ys_actual, ys_predicted):
    for i in range(columns_num):
        print('Variance of column #' + str(i + 1) + (' (index as in input data): %.10f' % column_vars[i]), \
              file=of, sep=config.output_separator)
    if normalize_by_mean_distance:
        for i in range(columns_num):
            print('Mean distance for column #' + str(i + 1) + (
            ' (index as in input data): %.10f' % column_mean_distances[i]), \
                  file=of, sep=config.output_separator)
    print('Column of y: %d' % (y_index + 1), file=of, sep=config.output_separator)
    print('Columns of xs: %s' % list_to_str(list(map(lambda x: x + 1, columns_xs)), left='', right='',
                                                       delim=', '), file=of)
    print('mean(abs(error)) = %.10f' % compute_mean_abs_error(ys_actual, ys_predicted), file=of,
          sep=config.output_separator)
    print('standard error = %.10f' % compute_standard_error(ys_actual, ys_predicted), file=of,
          sep=config.output_separator)

def print_test_output(row_id, row_to_predict, ys_to_try, best_rows_directions, best_shifts, best_func_deltas,
                      func_values, directions,
                      useCommaInNumbers, separator):
    filename = generate_output_filename(config.test_output_filename, row_id)
    file = open(filename, 'w')

    num_xs = row_to_predict.size - 1
    print(list_to_str(['point_y'] + ['point_x%d' % i for i in range(1, num_xs + 1)] +
                      ['delta'] +
                      ['point_direction_y'] + ['point_direction_x%d' % i for i in range(1, num_xs + 1)] +
                      ['shift_norm_y'] + ['shift_norm_x%d' % i for i in range(1, num_xs + 1)] +
                      ['func_value'] + ['direction'],
                      left='', right='', useCommaInNumbers=useCommaInNumbers, delim=separator), file=file)
    row_to_predict = row_to_predict.copy()
    for i in range(len(ys_to_try)):
        row_to_predict[0] = ys_to_try[i]
        best_rows_direction = best_rows_directions[:, i]
        shift = best_shifts[:, i]
        print(list_to_str(
            row_to_predict.tolist() +
            [best_func_deltas[i]] +
            best_rows_direction.tolist() +
            shift.tolist() +
            [func_values[i]] + [directions[i]],
            left='', right='', useCommaInNumbers=useCommaInNumbers, delim=separator), file=file)
    file.close()

########################################
# Math misc
########################################

def compute_mean_var(a):
    mean = float(sum(a)) / len(a)
    return (mean, sqrt(sum(map(lambda x: (x - mean)**2, a)) / len(a)))

def compute_mean(a):
    return compute_mean_var(a)[0]

def compute_var(a):
    return compute_mean_var(a)[1]

def reduce_data_set(rows_used, use_reduction_size, debug=True):
    n = len(rows_used)
    if debug:
        print('Rows used for learning before reduction: #' + str(n))
    rows_to_process = rows_used[:]
    rows_saved = []
    while len(rows_to_process) > 0:
        i = randint(0, len(rows_to_process) - 1)
        selected_row_id = rows_to_process[i]
        rows_saved.append(selected_row_id)

        selected_row_old_index = rows_used.index(selected_row_id)
        removed_region_begin = max(selected_row_old_index - use_reduction_size, 0)
        removed_region_end = min(selected_row_old_index + use_reduction_size, n - 1)
        for removed_index in range(removed_region_begin, removed_region_end + 1):
            try:
                rows_to_process.remove(rows_used[removed_index])
            except Exception:
                pass
    if debug:
        print('Rows used for learning after reduction: #' + str(len(rows_saved)))

    rows_saved.sort()
    return rows_saved

def normalize_data(data, normalizers):
    '''Normalizes data'''
    data /= normalizers

def scale_data(data, scalers):
    '''Normalizes data'''
    data *= np.array(scalers)

def compute_mean_abs_error(ys_actual, ys_predicted):
    try:
        return sum(map(lambda x: abs(x[0] - x[1]), zip(ys_actual, ys_predicted))) / len(ys_actual)
    except Exception:
        return -1.0

def compute_standard_error(ys_actual, ys_predicted):
    try:
        return (sum(map(lambda x: (x[0] - x[1])**2, zip(ys_actual, ys_predicted))) / len(ys_actual)) ** 0.5
    except Exception:
        return -1.0

def random_indeces(n, k):
    t = list(range(n))
    shuffle(t)
    return t[:k]

def shift_point_to_directions(initial_point, rows_directions):
    '''Returns two matrices, where each row represents a shift
    The first one is a matrix, where each row is a shifted point.
    The second one is a matrix, where each row is a vector of the shifts' projections on coordinates.'''
    num_directions = rows_directions.shape[0]
    dim = rows_directions.shape[1]
    shifted_points = np.zeros((num_directions, dim))
    shifts = np.zeros((num_directions, dim))
    for i in range(num_directions):
        dist = np.sqrt(np_compute_sqr_dist(initial_point, rows_directions[i, :]))
        shifts[i, :] = (rows_directions[i, :] - initial_point) / dist * config.unit_step
        shifted_points[i, :] = initial_point + shifts[i, :]
    return shifted_points, shifts

