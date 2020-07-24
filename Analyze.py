import numpy as np
from tensorflow import keras
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

FEATURES = ['Age', 'Games Played', 'Pro Bowl', 'All Pro',
            'Total TD', 'Fantasy Points', 'PPR',
            'DraftKings Points', 'FanDuel Points',
            'VBD', 'Position Rank', 'Overall Rank',
            'PPG', 'TD/G', 'Turnovers', 'Y/A', 'Points Scored']

metric = 'Fantasy Points'


def analyze(features, start_training, end_training, end_testing=2019):
    """
    Makes a Neural Network with the given features that outputs a projected value of a football players fantasy
    points scored next year based off of their stats this year.
    :param features: the list of features analyzed by the neural network.
    :param start_training: the starting year of the training.
    :param end_training: the ending year of the training.
    :param end_testing: the ending year of the testing.
    :return: Nothing.
    """

    # Get the training sets and test sets
    (x_train, y_train) = get_input_and_output(features, start_training, end_training)
    (x_test, y_test) = get_input_and_output(features, end_training, end_testing)

    # normalize the inputs
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # construct the model
    model = make_regression_model(len(features))

    # fit the model
    history = model.fit(x_train, y_train, batch_size=32, epochs=64, validation_split=0.2, verbose=0,
                        callbacks=[tfdocs.modeling.EpochDots()])

    # test_scores = model.evaluate(x_test, y_test, verbose=2)
    # print("Test loss:", str(test_scores[0])[:5])
    # print("Test accuracy:", str(test_scores[1] * 100)[:5], '%')

    # Graph the MAE of the tests
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylim([0, 100])
    label = 'MAE ' + '[' + metric + ']'
    plt.ylabel(label)
    plt.show()

    # Graph the true values vs predictions of the tests
    test_predictions = model.predict(x_test).flatten()
    a = plt.axes(aspect='equal')
    plt.scatter(y_test, test_predictions)
    xlabel = 'True Values ' + '[' + metric + ']'
    ylabel = 'Predictions ' + '[' + metric + ']'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    lims = [0, 450]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    # Graph the error for each test
    error = test_predictions - y_test
    plt.hist(error, bins=25)
    error_x_label = 'Prediction Error ' + '[' + metric + ']'
    plt.xlabel(error_x_label)
    _ = plt.ylabel("Count")
    plt.show()

    print()

    # Calculate and print the mean absolute percentage error
    mape = np.mean(np.abs((y_test - test_predictions) / y_test * 100))
    print('mape: ', str(mape)[:6], '%')

    # Calculate and print the mean absolute percentage error for projections over 100 points
    mape_projection_over_100 = []
    mape_actual_over_100 = []
    mape_no_outliers = []
    for i in range(len(y_test)):
        if y_test[i] > 100:
            mape_actual_over_100.append(abs((y_test[i] - test_predictions[i])) / y_test[i])
        if test_predictions[i] > 100:
            mape_projection_over_100.append(abs((y_test[i] - test_predictions[i])) / y_test[i])
        if abs(test_predictions[i] - y_test[i]) < 125:
            mape_no_outliers.append(abs((y_test[i] - test_predictions[i])) / y_test[i])

    mape_projection_over_100_val = sum(mape_projection_over_100) / len(mape_projection_over_100) * 100
    mape_actual_over_100_val = sum(mape_actual_over_100) / len(mape_actual_over_100) * 100
    mape_no_outliers_val = sum(mape_no_outliers) / len(mape_no_outliers) * 100

    print('mape (projections > 100): ', str(mape_projection_over_100_val)[:6], '%')
    print('mape (actual > 100): ', str(mape_actual_over_100_val)[:6], '%')
    print('mape (no outliers): ', str(mape_no_outliers_val)[:6], '%')


def get_input_and_output(features, start_year, end_year):
    input = []
    output = []

    for i in range(start_year, end_year):
        df = pd.read_csv('fantasy stats/fantasy-full' + str(i) + '.csv')
        next_df = pd.read_csv('fantasy stats/fantasy-full' + str(i + 1) + '.csv')
        names = next_df['Name'].tolist()

        for index, row in df.iterrows():
            if row['Name'] in names:
                values = []

                for feature in features:
                    if feature == 'PPG':
                        values.append(row['Fantasy Points'] / row['Games Played'])
                    elif feature == 'TD/G':
                        values.append(row['Total TD'] / row['Games Played'])
                    elif feature == 'Turnovers':
                        values.append(row['Fumbles'] + row['Interceptions'])
                    elif feature == 'Y/A':
                        if row['Pos'] == 'QB':
                            values.append(row['Passing Yards'] / row['Passing Attempts'])
                        elif row['Pos'] == 'RB':
                            values.append(row['Rushing Y/A'])
                        elif row['Pos'] == 'WR' or row['Pos'] == 'TE':
                            values.append(row['Y/R'])
                        else:
                            values.append(3)
                    elif feature == 'Points Scored':
                        values.append(7 * row['Total TD'] + 2 * row['2PM'] + 2 * row['2PP'])
                    else:
                        values.append(row[feature])

                input.append(values)

                index_of_name = names.index(row['Name'])

                next_metric = next_df.iloc[index_of_name][metric]
                output.append(next_metric)
                # curr_metric = row[metric]
                # metric_diff = next_metric - curr_metric
                # if metric_diff > 0:
                #     output.append(1)
                # else:
                #     output.append(0)

    return np.array(input), np.array(output)


def make_regression_model(input_length):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[input_length]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'accuracy'])

    return model


analyze(FEATURES, 2014, 2018)

# (x_train, y_train) = get_input_and_output(FEATURES, 2010, 2011)
# print(x_train[0])
