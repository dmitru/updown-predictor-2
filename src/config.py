__author__ = 'dmitry'

import utils

class Config:
    ids_rows_learn = []           # What data lines are used for prediction, ids of those lines
    rows_predict = []         # What data we need to predict
    prescalers = None         # What prescalers to use for each column
    columns_num = None
    columns_ys = None         # How many nearest points use for predictions
    output_filename = None
    normalizing = False       # Whether to normalize scalers during 'steps' optimization
    normalize_by_mean_distance = False
    extended = True
    config_filename = 'config.txt'  # From which file read the config
    input_filename = 'data.txt'
    input_separator = ' '     # Separates columns of data in input file
    output_separator = ' '
    output_numbers_use_comma = False
    output_print_header = True
    output_func_values = True
    output_weighted_sum_formula_results = True

    read_learn_data_from_file = False
    learn_data_filename = None

    use_reduction = None      # Whether to perform "stochastic reduction" of the set of used rows.
                              # It helps to randomly reduce the size of the dataset, increasing the speed of prediction.
    use_reduction_size = None # Parameter to "stochastic reduction". The greater the parameter, the fewer elements will
                              # remain in the dataset after reduction.
    reduced_rows_filename = None

    y_min = None              # See point 7 in 'TZ_1.txt'
    y_max = None              #
    y_step = None             #
    y_pow = None
    y_d = None

    debug = False

    compute_scalers = False
    unit_step = None
    directions_initial_num = None
    directions_increase = None
    directions_best_from_prev_step = None
    lowest_scaler_bound = None

    test_output = False
    test_output_filename = None
    use_squares_of_deltas = False

config = Config()

def read_config(fname):
    global config

    try:
        cf = open(fname)
    except IOError:
        utils.die("Can't open config file '" + fname + "'")

    params = {}
    cur_line = ""
    try:
        for l in cf:
            cur_line = l
            l = l.strip()
            if not l or l[0] == '#':
                continue
            d = l.split('=')
            if len(d) != 2:
                raise ValueError()

            param, value = map(lambda s: s.strip(), d)
            params[param] = value

            if param == 'learn' or param == 'predict' or param == 'columns_ys':
                value = map(lambda s: s.strip(), value.split(','))
                rows = []
                for v in value:
                    if v.isdigit():
                        rows.append(int(v))
                    else:
                        v = list(map(int, v.split(':')))
                        if len(v) != 2:
                            raise ValueError()
                        v[1] += 1
                        rows += range(*v)
                if param == 'predict':
                    config.rows_predict = rows
                elif param == 'learn':
                    config.ids_rows_learn = rows
                else: # param == 'columns_ys'
                    config.columns_ys = rows
            elif param == 'debug':
                if value.lower() == 'on':
                    config.debug = True
                elif value.lower() == 'off':
                    config.debug = False
                else:
                    raise ValueError()
            elif param == 'normalizing':
                if value.lower() == 'on':
                    config.normalizing = True
                elif value.lower() == 'off':
                    config.normalizing = False
                else:
                    raise ValueError()
            elif param == 'use_squares_of_deltas':
                if value.lower() == 'on':
                    config.use_squares_of_deltas = True
                elif value.lower() == 'off':
                    config.use_squares_of_deltas = False
                else:
                    raise ValueError()
            elif param == 'output_filename':
                config.output_filename = value
            elif param == 'reduced_rows_filename':
                config.reduced_rows_filename = value
            elif param == 'input_separator':
                config.input_separator = value
                if value == '' or value == 'space':
                    config.input_separator = ' '
                if value == 'tab':
                    config.input_separator = '\t'
            elif param == 'output_separator':
                config.output_separator = value
                if value == '' or value == 'space':
                    config.output_separator = ' '
                if value == 'tab':
                    config.output_separator = '\t'
            elif param == 'input_filename':
                config.input_filename = value
            elif param == 'learn_data_filename':
                config.learn_data_filename = value
            elif param == 'test_output_filename':
                config.test_output_filename = value
            elif param == 'columns_num':
                config.columns_num = int(value)
            elif param == 'prescalers':
                config.prescalers = list(map(lambda x: float(x.strip()), value.split(',')))
            elif param == 'y_min':
                config.y_min = float(value)
            elif param == 'y_max':
                config.y_max = float(value)
            elif param == 'y_d':
                config.y_d = float(value)
            elif param == 'y_pow':
                config.y_pow = float(value)
            elif param == 'y_step':
                config.y_step = float(value)
            elif param == 'use_reduction':
                if value.lower() == 'on':
                    config.use_reduction = True
                elif value.lower() == 'off':
                    config.use_reduction = False
                else:
                    raise ValueError()
            elif param == 'output_weighted_sum_formula_results':
                if value.lower() == 'on':
                    config.output_weighted_sum_formula_results = True
                elif value.lower() == 'off':
                    config.output_weighted_sum_formula_results = False
                else:
                    raise ValueError()
            elif param == 'output_func_values':
                if value.lower() == 'on':
                    config.output_func_values = True
                elif value.lower() == 'off':
                    config.output_func_values = False
                else:
                    raise ValueError()
            elif param == 'read_learn_data_from_file':
                if value.lower() == 'on':
                    config.read_learn_data_from_file = True
                elif value.lower() == 'off':
                    config.read_learn_data_from_file = False
                else:
                    raise ValueError()
            elif param == 'output_print_header':
                if value.lower() == 'on':
                    config.output_print_header = True
                elif value.lower() == 'off':
                    config.output_print_header = False
                else:
                    raise ValueError()
            elif param == 'test_output':
                if value.lower() == 'on':
                    config.test_output = True
                elif value.lower() == 'off':
                    config.test_output = False
                else:
                    raise ValueError()
            elif param == 'output_numbers_use_comma':
                if value.lower() == 'on':
                    config.output_numbers_use_comma = True
                elif value.lower() == 'off':
                    config.output_numbers_use_comma= False
                else:
                    raise ValueError()
            elif param == 'normalize_by_mean_distance':
                if value.lower() == 'on':
                    config.normalize_by_mean_distance = True
                elif value.lower() == 'off':
                    config.normalize_by_mean_distance= False
                else:
                    raise ValueError()
            elif param == 'extended':
                if value.lower() == 'on':
                    config.extended = True
                elif value.lower() == 'off':
                    config.extended = False
                else:
                    raise ValueError()
            elif param == 'use_reduction_size':
                config.use_reduction_size = int(value)
            elif param == 'use_ys_for_prediction':
                if value.lower() == 'on':
                    use_ys_for_prediction = True
                elif value.lower() == 'off':
                    use_ys_for_prediction = False
                else:
                    raise ValueError()
            elif param == 'lowest_scaler_bound':
                config.lowest_scaler_bound = float(value)
            elif param == 'directions_initial_num':
                config.directions_initial_num = int(value)
            elif param == 'directions_increase':
                config.directions_increase = float(value)
            elif param == 'directions_best_from_prev_step':
                config.directions_best_from_prev_step = int(value)
            elif param == 'unit_step':
                config.unit_step = float(value)
            elif param == 'compute_scalers':
                if value.lower() == 'on':
                    config.compute_scalers = True
                elif value.lower() == 'off':
                    config.compute_scalers = False
                else:
                    raise ValueError()
            else:
                raise ValueError()
    except ValueError:
        utils.die('Bad config line: ' + cur_line.strip())

    if config.columns_num is None:
        utils.die('\'columns_num\' is not specified in config file \'' + fname + '\'')
    if config.columns_num <= 0:
        utils.die('\'num_factors\' must be a positive integer')

    if config.compute_scalers:
        if config.unit_step is None:
            utils.die('\'unit_step\' is not specified in config file \'' + fname + '\', but option '
                                                                                   '\'compute_scalers\' is on')
        if config.lowest_scaler_bound is None:
            utils.die('\'lowest_scaler_bound\' is not specified in config file \'' + fname + '\', but option '
                                                                                   '\'compute_scalers\' is on')
        if config.directions_initial_num is None:
            utils.die('\'directions_initial_num\' is not specified in config file \'' + fname + '\', but option '
                                                                                   '\'compute_scalers\' is on')
        if config.directions_increase is None:
            utils.die('\'directions_increase\' is not specified in config file \'' + fname + '\', but option '
                                                                                   '\'compute_scalers\' is on')
        if config.directions_best_from_prev_step is None:
            utils.die('\'directions_best_from_prev_step\' is not specified in config file \'' + fname + '\', but option '
                                                                                   '\'compute_scalers\' is on')

    if config.use_reduction is None:
        utils.die('\'use_reduction\' is not specified in config file \'' + fname + '\'')
    if config.use_reduction_size is None:
        utils.die('\'use_reduction_size\' is not specified in config file \'' + fname + '\'')
    if config.use_reduction_size <= 0:
        utils.die('\'use_reduction_size\' must be a positive integer \'' + fname + '\'')

    if config.read_learn_data_from_file:
        if config.learn_data_filename is None or len(config.learn_data_filename) == 0:
            utils.die('\'learn_data_filename\' must be specified when \'read_learn_data_from_file\' is \'on\'')

    if config.y_min is None:
        utils.die('\'y_min\' is not specified')
    if config.y_max is None:
        utils.die('\'y_max\' is not specified')
    if config.y_d is None:
        utils.die('\'y_d\' is not specified')
    if config.y_step is None:
        utils.die('\'y_step\' is not specified')
    if config.y_pow is None:
        utils.die('\'y_pow\' is not specified')

    if len(config.rows_predict) == 0:
        utils.die('No rows specified in config file \'' + fname + '\'. Please, specify at least one row for prediction')

    if len(config.ids_rows_learn) == 0:
        utils.die('Warning: no working data range is specified explicitly; use all data lines by default')

    if config.prescalers is None:
        config.prescalers = [1.0 for _ in range(config.columns_num)]
    elif len(config.prescalers) < config.columns_num:
        utils.die('Length of \'prescalers\' list is less than \'columns_num\'')
    elif any(map(lambda x: x < 0, config.prescalers)):
        utils.die('\'prescalers\' must be a list of positive floating-point numbers')

    if len(set(config.ids_rows_learn).union(set(config.rows_predict))) != len(config.ids_rows_learn) + len(config.rows_predict):
        utils.die('rows_learn and rows_predict have overlapping common rows')

    config.columns_ys = list(map(lambda x: x - 1, config.columns_ys))

    cf.close()

    if config.debug:
        print('Parameters are:')
        for p in params.keys():
            print(p, '=', params[p])
        print()