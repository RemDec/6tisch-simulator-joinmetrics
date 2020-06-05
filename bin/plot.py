"""
Plot a stat over another stat.

Example:
    python plot.py --inputfolder simData/numMotes_50/ -x chargeConsumed --y aveLatency
"""
from __future__ import print_function

# =========================== imports =========================================

# standard
from builtins import range
import os
import argparse
import json
import glob
from collections import OrderedDict
import numpy as np

# third party
import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns  # NPEB_modif
import pandas as pd

# ============================ defines ========================================

KPIS = [
    'latency_max_s',
    'latency_avg_s',
    'latencies',
    'lifetime_AA_years',
    'sync_time_s',
    'join_time_s',
    'upstream_num_lost',
    'first_hop',  # vv NPEB_modif vv
    'joinRPL_time_s',
    'charge_synched',
    'charge_joined',
    'charge_joinedRPL',
    'charge_afterEB'
]

# ============================ main ===========================================

def main(options):

    # init
    data = OrderedDict()

    # chose lastest results
    subfolders = list(
        [os.path.join(options.inputfolder, x) for x in os.listdir(options.inputfolder)]
    )
    subfolder = max(subfolders, key=os.path.getmtime)

    phases_stats = {'times': {}, 'charges': {}}

    for key in options.kpis:
        # load data
        for file_path in sorted(glob.glob(os.path.join(subfolder, '*.kpi'))):
            curr_combination = os.path.basename(file_path)[:-8] # remove .dat.kpi
            with open(file_path, 'r') as f:

                # read kpi file
                kpis = json.load(f)

                # init data list
                data[curr_combination] = []

                # fill data list
                for run in kpis.values():
                    for mote in run.values():
                        if key in mote:
                            if mote[key] is not None:
                                data[curr_combination].append(mote[key])
        if key in ['sync_time_s', 'join_time_s', 'joinRPL_time_s']:
            phases_stats['times'][key] = data[data.keys()[0]]
        elif key in ['charge_synched', 'charge_joined', 'charge_joinedRPL', 'charge_afterEB']:
            phases_stats['charges'][key] = data[data.keys()[0]]
        # plot
        try:
            if key in ['lifetime_AA_years', 'latencies']:
                plot_cdf(data, key, subfolder)
            elif key == 'first_hop':  # NPEB_modif
                plot_histogram_hops(data, subfolder)
            elif key == 'charge_joined': # NPEB_modif
                plot_histogram_charge_joined(data, subfolder)
            else:
                plot_box(data, key, subfolder)

        except TypeError as e:
            print("Cannot create a plot for {0}: {1}.".format(key, e))
    plot_phase_times_boxplots(phases_stats['times'], subfolder)
    plot_phase_charges_boxplots(phases_stats['charges'], subfolder)
    print("Plots are saved in the {0} folder.".format(subfolder))

# =========================== helpers =========================================

def plot_phase_times_boxplots(phase_times, subfolder):
    stat_names = ['sync_time_s', 'join_time_s', 'joinRPL_time_s']
    labels = {stat_names[0]: 'Synchro',
              stat_names[1]: 'SecJoin',
              stat_names[2]: 'RPLjoin'}
    data = {'phase': [], 'value': []}
    for phase, vals in phase_times.items():
        data['phase'].extend([labels[phase]] * len(vals))
        data['value'].extend(vals)

    df = pd.DataFrame(data)
    ax = sns.boxplot(y='phase', x='value', data=data, orient='h', linewidth=1,
                     order=[labels[t] for t in stat_names[::-1]],
                     palette=['OrangeRed', 'Orange', 'Gold'],
                     notch=False, meanline=True, showmeans=True)
    ax.set_title("Time elapsed between steps of join process for all nodes")
    # x axis
    ax.set_xlabel("Time (s)")

    savefig(subfolder, "phase_times")
    plt.clf()


def plot_phase_charges_boxplots(phase_charges, subfolder):
    from SimEngine.Mote.tsch import DELAY_LOG_AFTER_EB
    stat_names = ['charge_synched', 'charge_joined', 'charge_joinedRPL', 'charge_afterEB']
    labels = {stat_names[0]: 'Synchro',
              stat_names[1]: 'SecJoin',
              stat_names[2]: 'RPLjoin',
              stat_names[3]: str(DELAY_LOG_AFTER_EB) + 's after EB'}
    data = {'phase': [], 'value': []}
    for phase, vals in phase_charges.items():
        data['phase'].extend([labels[phase]] * len(vals))
        data['value'].extend(map(lambda microC: microC/1000, vals))

    df = pd.DataFrame(data)
    ax = sns.boxplot(y='phase', x='value', data=data, orient='h', linewidth=1,
                     palette=['Tomato', 'OrangeRed', 'Orange', 'Gold'],
                     order=[labels[t] for t in stat_names[::-1]],
                     notch=False, meanline=True, showmeans=True)
    ax.set_title("Charge consumed between steps of join process for all nodes")
    # x axis
    ax.set_xlabel("Charge (mC)")

    savefig(subfolder, "phase_charges")
    plt.clf()


#NPEB_modif
def plot_histogram_hops(data, subfolder):
    for k, values in data.items():
        ax = sns.distplot(values, kde=False, hist_kws={"align": 'mid'})
        ax.set_title("Distribution of number of hops for the first data traffic of each node")
        ax.minorticks_on()
        # y axis
        ax.set_ylabel("Number of joined nodes")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
        # x axis
        ax.set_xticks(list(set(values)))
        ax.set_xlim(left=min(values)-0.5, right=max(values)+0.5)
        ax.xaxis.set_tick_params(which='minor', bottom=False)

        savefig(subfolder, "firsthop" + ".hist")
        plt.clf()


#NPEB_modif
def plot_histogram_charge_joined(data, subfolder):
    for k, values in data.items():
        values = map(lambda microC: microC/1000, values)
        ax = sns.distplot(values, bins=20, kde=False, hist_kws={"align": 'mid'}, rug=True)
        ax.set_title("Distribution of consumed charge values at joining achievement")
        ax.minorticks_on()
        # y axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_tick_params(which='minor', bottom=False)
        ax.grid(axis="y")
        # x axis
        ax.set_xlabel("Charge value (mC)")

        savefig(subfolder, "chargejoined" + ".hist")
        plt.clf()


def plot_cdf(data, key, subfolder):
    for k, values in data.items():
        # convert list of list to list
        if type(values[0]) == list:
            values = sum(values, [])

        values = [None if value == 'N/A' else value for value in values]
        # compute CDF
        sorted_data = np.sort(values)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, label=k)

    plt.xlabel(key)
    plt.ylabel("CDF")
    plt.legend()
    savefig(subfolder, key + ".cdf")
    plt.clf()

def plot_box(data, key, subfolder):
    plt.boxplot(list(data.values()))
    plt.xticks(list(range(1, len(data) + 1)), list(data.keys()))
    plt.ylabel(key)
    savefig(subfolder, key)
    plt.clf()

def savefig(output_folder, output_name, output_format="png"):
    # check if output folder exists and create it if not
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # save the figure
    plt.savefig(
        os.path.join(output_folder, output_name + "." + output_format),
        bbox_inches     = 'tight',
        pad_inches      = 0,
        format          = output_format,
    )

def parse_args():
    # parse options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputfolder',
        help       = 'The simulation result folder.',
        default    = 'simData',
    )
    parser.add_argument(
        '-k','--kpis',
        help       = 'The kpis to plot',
        type       = list,
        default    = KPIS
    )
    parser.add_argument(
        '--xlabel',
        help       = 'The x-axis label',
        type       = str,
        default    = None,
    )
    parser.add_argument(
        '--ylabel',
        help       = 'The y-axis label',
        type       = str,
        default    = None,
    )
    parser.add_argument(
        '--show',
        help       = 'Show the plots.',
        action     = 'store_true',
        default    = None,
    )
    return parser.parse_args()

if __name__ == '__main__':

    options = parse_args()

    main(options)