import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
from sympy.parsing.sympy_parser import parse_expr
import os
import lca_algebraic as agb


def plot_lca(results_df, relative=False, custom_units=None):
    if '*sum*' in results_df.index:
        df_to_plot = results_df.drop('*sum*', axis=0).transpose()
    else:
        df_to_plot = results_df.transpose()
    if relative:
        df_to_plot = df_to_plot.div(df_to_plot.sum(1),axis=0)

    # Left Y-axis ticks
    ind_clean = [ind[0].upper() + ind[1:].split('- ')[0] for ind in df_to_plot.index]
    df_to_plot.index = ind_clean

    # Plot bar chart
    ax = df_to_plot.plot(kind='barh',
                         stacked=True,
                         figsize=(11,6),
                         mark_right=True,
                         cmap='tab20',
                         #xlim=(0,1)
    )

    # Right Y-axis ticks
    ax_values = ax.twinx()
    ax_values.set_ylim(ax.get_ylim())
    if '*sum*' in results_df.index:
        values = results_df.loc["*sum*"].values.tolist()
    elif len(results_df.index) == 1:
        values = results_df.values[0].tolist()
    else:
        values = ['' for i in range(len(ind_clean()))]

    if custom_units:
        if isinstance(custom_units,str):
            units = [custom_units for i in range(len(ind_clean))] 
        elif isinstance(custom_units,list) and len(custom_units)==1:
            units = [custom_units[0] for i in range(len(ind_clean))]
        elif isinstance(custom_units,list) and len(custom_units)==len(ind_clean):
            units=custom_units
    else:
        units = ['[' + ind.split('[')[1] for ind in results_df.transpose().index]
    ax_values.set_yticks(np.arange(len(results_df.columns)),labels=[str("{:.2e}".format(m)) + " " + n for m,n in zip(values,units)])

    # X-tickz
    if relative:
        ax.set_xlim(0,1)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        #ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%']);
    
    # Legend
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.1),
              fancybox=True,
              shadow=False,
              ncol=4)
    return ax


def plot_lca_comparison(results_dfs):
    if not isinstance(results_dfs, list):
        results_dfs = [results_dfs]

    for i, df in enumerate(results_dfs):
        if '*sum*' in df.index:
            results_dfs[i] = df.loc['*sum*'].transpose().rename('Product ' + chr(ord('@')+(i+1)))
        else:
            results_dfs[i] = df.transpose()
        
    df_to_plot = pd.concat(results_dfs, axis=1)
    df_to_plot=df_to_plot.div(df_to_plot.iloc[:,0], axis=0)
    units = ['[' + ind.split('[')[1] for ind in df_to_plot.index]
    ind_clean = [ind[0].upper() + ind[1:].split('- ')[0] for ind in df_to_plot.index]
    df_to_plot.index = ind_clean
    
    # Plot bar chart
    ax = df_to_plot.plot(kind='barh',
                         stacked=False,
                         figsize=(11,8),
                         mark_right=True,
                         #cmap='tab20',
                         #xlim=(0,1)
                        )

    # Reference dashed line at x=1.0
    ax.axvline(1.0, linestyle='--')
    
    # Right Y-axis ticks
    ax_values = ax.twinx()
    ax_values.set_ylim(ax.get_ylim())
    ax_values.set_yticks(np.arange(len(df_to_plot.index)),labels=units)
    
    # Legend
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.1),
              fancybox=True,
              shadow=False,
              ncol=4)

    ax.set_title('LCA results (relative comparison)')
    return ax


def print_parameters(db_name: str = None):
    dict = {"name": [], "description": [], "values": [], "default": [], "unit": []}
    for e in agb.params._listParams(db_name):
        dict["name"].append(e.name)
        dict["description"].append(e.description)
        dict["values"].append(e.values if e.type == 'enum' else "float")
        dict["default"].append(e.default)
        dict["unit"].append(e.unit)
    return pd.DataFrame(dict)


def recursive_activities(user_db: str, act):
    """Traverse tree of sub-activities of a given activity, until background database is reached."""
    activities = []
    units = []
    locations = []
    parents = []
    exchanges = []
    levels = []
    dbs = []

    def _recursive_activities(act,
                              activities, units, locations, parents, exchanges, levels, dbs,
                              parent: str = "", exc: dict = {}, level: int = 0):

        name = act.as_dict()['name']
        unit = act.as_dict()['unit']
        loc = act.as_dict()['location']
        exchange = _getAmountOrFormula(exc)
        db = act.as_dict()['database']
        if loc != 'GLO':
            name += f' [{loc}]'

        # to stop BEFORE reaching the first level of background activities
        # if db != USER_DB:  # to stop BEFORE reaching the first level of background activities
        #    return

        activities.append(name)
        units.append(unit)
        locations.append(loc)
        parents.append(parent)
        exchanges.append(exchange)
        levels.append(level)
        dbs.append(db)

        # to stop AFTER reaching the first level of background activities
        if db != user_db:
            return

        for exc in act.technosphere():
            _recursive_activities(exc.input, activities, units, locations, parents, exchanges, levels, dbs,
                                  parent=name,
                                  exc=exc,
                                  level=level + 1)
        return

    def _getAmountOrFormula(ex):
        """ Return either a fixed float value or an expression for the amount of this exchange"""
        if 'formula' in ex:
            return parse_expr(ex['formula'])
        elif 'amount' in ex:
            return ex['amount']
        return ""

    _recursive_activities(act, activities, units, locations, parents, exchanges, levels, dbs)
    data = {'activity': activities,
            'unit': units,
            'location': locations,
            'level': levels,
            'database': dbs,
            'parent': parents,
            'exchange': exchanges}
    df = pd.DataFrame(data, index=activities)

    df['description'] = df['activity'] + "\n (" + df['unit'] + ")"

    return df


def graph_activities(user_db: str, model, network_file_path : str):
    """
    Plots an interactive tree to visualize the activities and exchanges declared in the LCA module.
    """

    # Get LCA activities
    df = recursive_activities(user_db, model)

    net = Network(notebook=True, directed=True, layout=True, cdn_resources='remote')

    activities = df['activity']
    descriptions = df['description']
    parents = df['parent']
    amounts = df['exchange']
    levels = df['level']
    dbs = df['database']

    edge_data = zip(activities, descriptions, parents, amounts, levels, dbs)

    for e in edge_data:
        src = e[0]
        desc = e[1]
        dst = e[2]
        w = e[3]
        n = e[4]
        db = e[5]

        color = '#97c2fc' if db == user_db else 'lightgrey'
        if dst == "":
            net.add_node(src, desc, title=src, level=n + 1, shape='box', color=color)
            continue
        net.add_node(src, desc, title=src, level=n + 1, shape='box', color=color)
        net.add_node(dst, dst, title=dst, level=n, shape='box')
        net.add_edge(src, dst, label=str(w))

    net.set_edge_smooth('vertical')
    net.toggle_physics(False)
    
    # The drawback of this method is that the creation of directories and file is not controled,
    # it is created based on the working directory. So we will have to do some shenanigans to get
    # and change the working directory once we are done with the creation of the graph.

    directory_to_save_graph = os.path.dirname(network_file_path)
    graph_name = os.path.basename(network_file_path)
    old_working_directory = os.getcwd()

    # Set the new working directory
    os.chdir(directory_to_save_graph)
    
    net.show(graph_name)

    # Change the working directory back
    os.chdir(old_working_directory)

    return net


def lca_monte_carlo(model, methods, n_runs, cfs_uncertainty: bool = False, **params,):
    """
    Run Monte Carlo simulations to assess uncertainty on the impact categories.
    Input uncertainties are embedded in EcoInvent activities.
    Parameters used in the parametric study are frozen.
    """

    if not isinstance(methods, list):
        methods = [methods]

    # Freeze params
    db = model[0]  # get database in which model is defined
    if agb.helpers._isForeground(db):
        agb.freezeParams(db, **params)  # freeze parameters

    # Monte Carlo for each impact category with vanilla brightway
    scores_dict = {}
    functional_unit = {model: 1}

    if cfs_uncertainty:  # uncertainty on impact methods --> MC must be run for each impact
        for method in methods:
            print("### Running Monte Carlo for method " + str(method) + " ###")
            mc = bw.MonteCarloLCA(functional_unit, method)  # MC on inventory uncertainties (background db)
            scores = [next(mc) for _ in range(n_runs)]
            scores_dict[method] = scores

    else:  # TODO: automatically detect if impact method contains uncertain characterization factors
        def multiImpactMonteCarloLCA(functional_unit, list_methods, iterations):
            """
            https://github.com/maximikos/Brightway2_Intro/blob/master/BW2_tutorial.ipynb
            """
            # Step 1
            MC_lca = bw.MonteCarloLCA(functional_unit)
            MC_lca.lci()
            # Step 2
            C_matrices = {}
            scores_dict = {}
            # Step 3
            for method in list_methods:
                MC_lca.switch_method(method)
                C_matrices[method] = MC_lca.characterization_matrix
                scores_dict[method] = []
            # Step 4
            #results = np.empty((len(list_methods), iterations))
            # Step 5
            for iteration in range(iterations):
                next(MC_lca)
                for method_index, method in enumerate(list_methods):
                    score = (C_matrices[method] * MC_lca.inventory).sum()
                    # results[method_index, iteration] = score
                    scores_dict[method].append(score)
            return scores_dict

        print("### Running Multi-Impacts Monte Carlo (warning: uncertainty restricted to LCI) ###")
        scores_dict = multiImpactMonteCarloLCA(functional_unit, methods, n_runs)

    df = pd.DataFrame.from_dict(scores_dict, orient='columns')

    return df