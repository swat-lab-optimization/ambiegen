"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for processing the experimental results
"""

import os
import argparse
import csv
from itertools import combinations
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Tuple, Dict, Any
import logging as log
from scipy.stats import mannwhitneyu
from ambiegen.generators.abstract_generator import AbstractGenerator
from ambiegen.generators.kappa_generator import KappaRoadGenerator
from ambiegen.generators.obstacle_generator import ObstacleGenerator
from ambiegen.test_generators.lkas_test_generator import LKASTestGenerator
from matplotlib.ticker import MaxNLocator
from aerialist.px4.obstacle import Obstacle
from ambiegen.common.cliffsDelta import cliffsDelta
import time



def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )
    log.info(start_msg)


def parse_arguments():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description="A tool for generating test cases for autonomous systems",
        epilog="For more information, please visit ",
    )
    parser.add_argument(
        "--stats-path",
        nargs="+",
        help="The source folders of the metadate to analyze",
        required=True,
    )
    parser.add_argument(
        "--stats-names",
        nargs="+",
        help="The names of the corresponding algorithms",
        required=True,
    )
    parser.add_argument(
        "--plot-name", help="Name to add to the plots", required=False, default="", type=str
    )
    parser.add_argument(
        "--problem", help="Type of the problem to analyze. Available options: [ads, uav]", required=False, default="ads", type=str
    )
    in_arguments = parser.parse_args()
    return in_arguments


def build_median_table(
    fitness_list, diversity_list, column_names, plot_name, save_dir="stats"
):
    """
    The function `build_median_table` takes in fitness and diversity lists, column names, and a plot
    name, and creates a table with mean fitness and diversity values, as well as p-values and effect
    sizes if the lists have two elements each, and writes the table to a CSV file.

    Args:
      fitness_list: The `fitness_list` parameter is a list of lists, where each inner list represents
    the fitness values for a particular algorithm. Each inner list should contain the fitness values for
    a specific algorithm.
      diversity_list: The `diversity_list` parameter is a list of lists, where each inner list
    represents the diversity values for a specific algorithm. Each inner list should contain the
    diversity values for that algorithm.
      column_names: The parameter "column_names" is a list of names for the columns in the table. It is
    used to label the different algorithms or methods being compared in the table.
      plot_name: The `plot_name` parameter is a string that represents the name of the plot or table
    that will be generated. It will be used to create a CSV file with the results of the function.
    """
    columns = ["Metric"]
    for name in column_names:
        columns.append(name)

    row_0 = ["Failure number"]
    for alg in fitness_list:
        row_0.append(round(np.mean(alg), 3))

    row_1 = ["Mean sparseness"]
    for alg in diversity_list:
        row_1.append(round(np.mean(alg), 3))

    if (len(fitness_list) == 2) and (len(diversity_list) == 2):
        row_0.append(
            round(
                mannwhitneyu(fitness_list[1], fitness_list[0], alternative="two-sided")[
                    1
                ],
                3,
            )
        )
        row_0.append(round(cliffsDelta(fitness_list[1], fitness_list[0])[0], 3))
        row_0.append(cliffsDelta(fitness_list[1], fitness_list[0])[1])

        row_1.append(
            round(
                mannwhitneyu(
                    diversity_list[0], diversity_list[1], alternative="two-sided"
                )[1],
                3,
            )
        )
        row_1.append(round(cliffsDelta(diversity_list[0], diversity_list[1])[0], 3))
        row_1.append(cliffsDelta(diversity_list[0], diversity_list[1])[1])
        columns.append("p-value")
        columns.append("Effect size")

    rows = [columns, row_0, row_1]

    log.info(f"Writing results to {plot_name}_res.csv")
    save_dir = save_dir + "_" + plot_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, plot_name + "_res.csv"), "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def build_cliff_data(
    fitness_list, diversity_list, column_names, plot_name, save_dir="stats"
):
    """
    The function `build_cliff_data` takes in fitness and diversity lists, column names, and a plot name,
    and writes the calculated p-values and effect sizes to separate CSV files for fitness and diversity.

    Args:
      fitness_list: The `fitness_list` parameter is a list of fitness values for different pairs of
    data. Each element in the list represents the fitness values for a specific pair of data.
      diversity_list: The `diversity_list` parameter is a list of lists, where each inner list
    represents the diversity values for a specific column or feature. Each inner list should contain the
    diversity values for that column or feature.
      column_names: The `column_names` parameter is a list of column names. It represents the names of
    the columns in the data that you want to analyze.
      plot_name: The `plot_name` parameter is a string that represents the name of the plot or data file
    that will be generated. It is used to create the output file names by appending
    "_res_p_value_fitness.csv" and "_res_p_value_diversity.csv" respectively.
    """
    title = ["A", "B", "p-value", "Effect size"]
    rows = [title]
    for pair in combinations(range(0, len(fitness_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                fitness_list[pair[0]], fitness_list[pair[1]], alternative="two-sided"
            )[1]
        )
        delta_value = round(
            cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info("Writing cliff delta data to file: " + plot_name + "_res_p_value.csv")

    save_dir = save_dir + "_" + plot_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, plot_name + "_res_p_value_failures.csv"), "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    rows = [title]
    for pair in combinations(range(0, len(diversity_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                diversity_list[pair[0]],
                diversity_list[pair[1]],
                alternative="two-sided",
            )[1]
        )

        delta_value = round(
            cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info(f"Writing to {plot_name + '_res_p_value_diversity.csv'}")

    with open(
        os.path.join(save_dir, plot_name + "_res_p_value_diversity.csv"),
        "w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def plot_convergence(dfs, stats_names, plot_name):
    """
    Function for plotting the convergence of the algorithms
    It takes a list of dataframes and a list of names for the dataframes, and plots the mean and
    standard deviation of the dataframes

    :param dfs: a list of dataframes, each containing the mean and standard deviation of the fitness of
    the population at each generation
    :param stats_names: The names of the algorithms
    """
    fig, ax = plt.subplots()

    plt.xlabel("Number of simulations", fontsize=16)
    plt.ylabel("Failures", fontsize=16)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()

    len_df = np.inf
    for i, df in enumerate(dfs):
        cur_len = len(dfs[i]["mean"])
        if cur_len < len_df:
            len_df = cur_len

    for i, df in enumerate(dfs):
        # x = np.arange(0, len(dfs[i]["mean"]))
        x = np.arange(0, len_df)
        plt.plot(x, dfs[i]["mean"][:len_df], label=stats_names[i])
        plt.fill_between(
            x,
            np.array(dfs[i]["mean"][:len_df] - dfs[i]["std"][:len_df]),
            np.array(dfs[i]["mean"][:len_df] + dfs[i]["std"][:len_df]),
            alpha=0.2,
        )
        plt.legend()

    log.info("Saving plot to " + plot_name + "_convergence.png")
    plt.savefig(plot_name + "_convergence.png", bbox_inches="tight")
    plt.close()


def calculate_test_list_novelty(
    test_list: list, generator: AbstractGenerator, out=False
) -> np.ndarray:
    """
    Calculate the novelty of a test list.

    This function calculates the novelty of a given test list by comparing each pair of tests
    in the list and calculating the novelty score using the `calc_novelty` function.

    Parameters:
    - test_list (list): A list of test objects.
    - problem (str): The problem type. Default is "robot".

    Returns:
    - novelty (float): The average novelty score of the test list.
    """

    all_novelty = []
    for i in range(len(test_list)):
        local_novelty = []
        for j in range(i + 1, len(test_list)):
            current1 = test_list[i]  # res.history[gen].pop.get("X")[i[0]]
            current2 = test_list[j]  # res.history[gen].pop.get("X")[i[1]]
            if out:
                nov = generator.cmp_out_func(current1, current2)
            else:
                nov = generator.cmp_func(current1, current2)
            # print("Novelty", nov)
            local_novelty.append(nov)
        if local_novelty:
            all_novelty.append(sum(local_novelty) / len(local_novelty))
            # all_novelty.append(max(local_novelty))
            # all_novelty.append(min(local_novelty))

    return all_novelty


def plot_boxplot(
    data_list, label_list, name, max_range=None, plot_name="", save_dir="boxplots"
):
    """
     Function for plotting the boxplot of the statistics of the algorithms
    It takes a list of lists, a list of labels, a name, and a max range, and plots a boxplot of the data

    :param data_list: a list of lists, each list containing the data for a particular algorithm
    :param label_list: a list of labels, each label corresponding to the data in the data_list
    :param name: the name of the plot
    :param max_range: the maximum value of the y-axis
    """

    fig, ax1 = plt.subplots(figsize=(12, 4))  # figsize=(8, 4)
    ax1.set_xlabel("Algorithm", fontsize=20)
    # ax1.set_xlabel("Rho value", fontsize=20)
    ax1.set_ylabel(name, fontsize=20)

    ax1.tick_params(axis="both", labelsize=18)

    ax1.yaxis.grid(
        True, linestyle="-", which="both", color="darkgray", linewidth=2, alpha=0.5
    )
    if max_range == None:
        max_vals = [max(x) for x in data_list]
        max_range = max(max_vals) + 0.1 * max(max_vals)
        # max_range = 110
        # max_range = max(data_list) + 0.1*max(data_list)
    top = max_range
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.boxplot(data_list, widths=0.45, labels=label_list)

    plt.subplots_adjust(bottom=0.15, left=0.16)

    save_dir = save_dir + "_" + plot_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig.savefig(
        os.path.join(save_dir, plot_name + "_" + name + ".png"), bbox_inches="tight"
    )
    plt.close()
    # log.info(f"Saving box plot: {os.path.join(save_dir, plot_name + "_" + name + ".png")}")

def analyse(stats_path, stats_names, plot_name):
    """
    Main function for building plots comparing the algorithms
    It takes a list of paths to folders containing the results of the tool runs, and a list of names
    of the runs, and it plots the convergence and the boxplots of the fitness and novelty

    :param stats_path: a list of paths to the folders containing the stats files
    :param stats_names: list of strings, names of the runs
    """
    convergence_paths = []
    stats_paths = []
    conv_flag = False
    for path in stats_path:
        for file in os.listdir(path):
            if "conv" in file:
                convergence_paths.append(os.path.join(path, file))
                conv_flag = True
            if "stats" in file:
                stats_paths.append(os.path.join(path, file))

    if conv_flag:
        dfs = {}
        for i, file in enumerate(convergence_paths):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            dfs[i] = pd.DataFrame(data=data)
            dfs[i]["mean"] = dfs[i].mean(axis=1)
            dfs[i]["std"] = dfs[i].std(axis=1)

        plot_convergence(dfs, stats_names, plot_name+"_fitness")

    fitness_list = []
    best_fitness_list = []
    novelty_list = []
    time_list = []
    max_fitness = 0
    for i, file in enumerate(stats_paths):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_fitness = []
        results_novelty = []
        results_time = []
        results_best_fitness = []

        for m in range(len(data)):
            fitness_data = [abs(d) for d in data["run" + str(m)]["fitness"]]
            value = max(fitness_data)
            if value > max_fitness:
                max_fitness = value
            best_fitness = value
            results_fitness.extend(fitness_data)  # data["run"+str(m)]["fitness"]
            results_novelty.append(data["run" + str(m)]["novelty"])
            results_best_fitness.append(best_fitness)

            if "times" in str(data):
                results_time.extend(data["run" + str(m)]["times"])

        fitness_list.append(results_fitness)
        novelty_list.append(results_novelty)
        time_list.append(results_time)
        best_fitness_list.append(results_best_fitness)

    if results_time:
        max_time = max(max(time_list[0]), max(time_list[1]))
        plot_boxplot(time_list, stats_names, "Time, s", max_time + 0.2, plot_name)
        build_times_table(time_list, stats_names)

    plot_boxplot(
        fitness_list, stats_names, "Fitness", max_fitness + 3, plot_name
    )  # + 2
    plot_boxplot(novelty_list, stats_names, "Diversity", 1.05, plot_name)

    build_median_table(fitness_list, novelty_list, stats_names, plot_name)
    build_cliff_data(fitness_list, novelty_list, stats_names, plot_name)

    compare_mean_best_values_found(best_fitness_list, stats_names, plot_name)
    compare_p_val_best_values_found(best_fitness_list, stats_names, plot_name)


if __name__ == "__main__":
    arguments = parse_arguments()
    setup_logging("log.txt", False)

    stats_path = arguments.stats_path
    stats_names = arguments.stats_names
    plot_name = arguments.plot_name
    problem = arguments.problem

    if problem == "uav":
        analyse_uav_tests(stats_path, stats_names, plot_name)
    elif problem == "ads":
        analyse_ads_tests(stats_path, stats_names, plot_name)
    else:
        log.error("Invalid problem type. Please specify either 'ads' or 'uav'.")
