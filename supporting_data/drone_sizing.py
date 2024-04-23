import numpy as np
from scipy.optimize import fmin_slsqp
import lca_algebraic as agb
import pandas as pd


def drone_sizing(performances, technologies, param_acv, lcia_methods_dict, variables_conception):
    """
    Modèle pour évaluer le dimensionnement du drone et de ses composants. 
    On se place dans le cas du vol stationnaire. 
    Il faut alors dimensionner les composants pour fournir suffisament de puissance (contrebalancer le poids), suffisament longtemps (durée de la mission).
    Ce modèle doit être appelé dans un problème d'optimisation pour assurer le respect des contraintes et la minimisation de l'objectif (masse ou impact environnemental).
    """
    
    ### Performances requises
    M_load = performances[0]  # [kg] charge à transporter
    t_mission = performances[1]  # [min] durée du vol requise
    
    ### Technologies de références
    propeller_mass_diameter_ratio = 0.68  # [kg/m^3] paramètre pour relier la masse d'une hélice à son diamètre
    motor_mass_torque_ratio = 0.28  # [kg/(N.m)^(3/3.5)] paramètre pour relier la masse d'un moteur à son couple nominal
    battery_specific_energy = technologies[0]  # [Wh/kg] rapport énergie/masse de la batterie
    battery_specific_power = 5000  # [W/kg] rapport puissance/masse de la batterie
    material_density = 1700.0  # [kg/m3] densité du matériau pour la structure
    
    
    ### Dimensionnement des composants à partir des variables de conception
    # Hélices
    D_propeller = variables_conception[0] # [m] diamètre hélices
    N_propeller = 4  # [-] nombre d'hélices
    M_propeller = N_propeller * propeller_mass_diameter_ratio * D_propeller ** 3  # [kg] estimation de la masse des hélices
    
    # Moteurs
    T_motor = variables_conception[1] # [N.m] couple nominal que peuvent fournir les moteurs
    N_motor = N_propeller  # il y a un moteur par hélice
    M_motor = N_motor * motor_mass_torque_ratio * T_motor ** (3/3.5)  # [kg] estimation de la masse des moteurs
    
    # Batterie
    E_battery = variables_conception[2]  # [Wh] energie contenue dans la batterie
    M_battery = E_battery / battery_specific_energy  # [kg] estimation de la masse de la batterie
    P_battery = M_battery * battery_specific_power  # [W] estimation de la puissance de la batterie
    
    # Structure
    L_arm = variables_conception[3]  # [m] longueur des bras
    N_arm = N_propeller  # [-] un bras par hélice
    D_in = 0.05  # [m] diamètre interne des bras (tubes)
    D_out = 0.051  # [m] diamètre externe des bras (tubes)
    M_arm = N_arm * np.pi / 4 * (D_out**2 - D_in**2) * L_arm * material_density  # [kg] masse des bras
    M_body = 1.5 * M_arm  # [kg] estimation de la masse du corps central
    M_structure = M_arm + M_body  # [kg] masse de structure
    
    # Masse totale
    M_drone = M_load + M_propeller +  M_motor + M_battery + M_structure
    
    
    ### Evaluation des performances en vol stationnaire
    # Hélices
    rho_air = 1.225  # [kg/m^3] densité de l'air
    Ct = 0.1  # [-] coefficient de poussée
    Cp = 0.04  # [-] coefficient de puissance
    F_propeller = M_drone * 9.81 / N_propeller  # [N] poussée à fournir (par hélice)
    n_propeller = (F_propeller / (rho_air * Ct * D_propeller**4)) ** (1/2)  # [Hz] vitesse de rotation de l'hélice
    P_propeller = rho_air * Cp * n_propeller**3 * D_propeller**5  # [W] puissance de l'hélice
    T_propeller = P_propeller / (2 * np.pi * n_propeller)  # [N.m] couple minimal que le moteur doit être capable de fournir à l'hélice
    
    # Moteurs
    n_motor = n_propeller  # le moteur tourne à la même vitesse que l'hélice
    eta_motor = 0.9  # [-] rendement du moteur (considéré constant ici)
    P_motor = T_motor * (2 * np.pi * n_motor) * eta_motor  # [W] puissance électrique que la batterie doit être capable de fournir au moteur
    
    # Energie consommée sur la mission
    E_mission = N_motor * P_motor * (t_mission / 60)  # [Wh] énergie requise pour compléter la mission
    
    
    ### ACV
    lca_parameters = dict(
        n_missions=param_acv["n_missions"],
        elec_mix=param_acv["elec_mix"],        
        mass_propellers=M_propeller,
        mass_motors=M_motor,
        mass_structure=M_structure,
        mass_batteries=M_battery,
        battery_type=param_acv["battery_type"],
        n_cycles_battery=param_acv["n_cycles_battery"],
        mission_energy=E_mission/1000,  # [kWh] --> diviser par 1000
    )
    lca_model = param_acv['lca_model']

    if lcia_methods_dict:
        lcia_methods, lcia_weights = list(lcia_methods_dict.keys()), list(lcia_methods_dict.values())
        lca_results = agb.compute_impacts(
            lca_model,
            lcia_methods,
            **lca_parameters
        ).iloc[0].values
        lca_score = sum(impact_score * weight for impact_score, weight in zip(lca_results, lcia_weights))
    else:
        lca_score = None
    
    
    ### Contraintes et objectif d'optimisation
    # Contraintes
    cnstr_couple = (T_motor - T_propeller) / T_motor  # contrainte pour s'assurer que les moteurs peuvent fournir le couple nécessaire
    cnstr_puissance = (P_battery - N_motor * P_motor) / P_battery  # contrainte pour s'assurer que la batterie est suffisant puissante
    cnstr_bras = (L_arm - (D_propeller / 2) / (np.sin(np.pi / N_arm))) / L_arm  # contrainte sur la longueur des bras pour que les hélices ne se touchent pas
    cnstr_energie = (0.8 * E_battery - E_mission) / E_battery  # contrainte pour s'assurer que la batterie a suffisamment d'énergie pour conclure la mission (0.8 pour marge)
    constraints = [cnstr_couple, cnstr_puissance, cnstr_bras, cnstr_energie]
    
    # Fonction objectif à minimiser
    obj_mass = M_drone  # minimisation de la masse du drone
    obj_energy = E_mission  # minimisation de l'énergie consommée
    obj_lca = lca_score  # minimisation de l'impact environnemental
    objectives = [obj_mass, obj_energy, obj_lca]
    
    return constraints, objectives, lca_parameters


def sizing_optimization(performances, technologies, param_acv, objectif, accuracy: float = 1e-9):

    lcia_methods_dict = {}
    
    # Objectif valide ?
    if not isinstance(objectif, dict):
        if objectif not in {'mass', 'energy'}:
            if objectif not in agb.findMethods(""):
                raise ValueError("Objective must be one of %r or a set of valid LCIA methods (see agb.findMethods())." % {'mass', 'energy'})
            else:
                lcia_methods_dict = {objectif: 1.0}  # attribute a weight of 100% to the single LCIA method provided
    else:
        for key in objectif.keys():
            if key not in agb.findMethods(""):
                raise ValueError("Objective must be one of %r or a set of valid LCIA methods (see agb.findMethods())." % {'mass', 'energy'})
        lcia_methods_dict = objectif

    # Point de départ pour les variables de conception
    initial_guess = [
        0.5,    # [m] D_propeller
        1.0,   # [N.m] T_motor
        100.0,   # [Wh] E_battery
        0.5,    # [m] L_arm
    ]
    
    # Bornes sur les variables de conception
    bounds = [
        (0.1, 1.0),     # [m] D_propeller
        (0.01, 10.0),   # [N.m] T_motor
        (10.0, 1000.0), # [Wh] E_battery
        (0.1, 1.0),     # [m] L_arm
    ]
    
    # Contraintes d'optimisation
    contraintes = lambda x: drone_sizing(performances, technologies, param_acv, lcia_methods_dict, x)[0]
    
    # Objectif d'optimisation
    if objectif == 'mass':
        scaler_value = drone_sizing(performances, technologies, param_acv, lcia_methods_dict, initial_guess)[1][0]
        obj_func = lambda x: drone_sizing(performances, technologies, param_acv, lcia_methods_dict, x)[1][0] / scaler_value  # Ensure value close to unity for better convergence
    elif objectif == 'energy':
        scaler_value = drone_sizing(performances, technologies, param_acv, lcia_methods_dict, initial_guess)[1][1]
        obj_func = lambda x: drone_sizing(performances, technologies, param_acv, lcia_methods_dict, x)[1][1] / scaler_value
    else:
        scaler_value = drone_sizing(performances, technologies, param_acv, lcia_methods_dict, initial_guess)[1][2]
        obj_func = lambda x: drone_sizing(performances, technologies, param_acv, lcia_methods_dict, x)[1][2] / scaler_value

    # Routine d'optimisation
    variables_conception_opt = fmin_slsqp(  # SLSQP est un algorithme d'optimisation à descente de gradient
        func=obj_func,  # fonction objectif à minimiser
        x0=initial_guess,  # point de départ de l'algorithme
        bounds=bounds,  # bornes sur les variables de conception
        f_ieqcons=contraintes,  # contraintes d'optimisation
        iter=1500,  # nombre max d'itérations
        acc=accuracy,  # critère d'arrêt de l'optimisation (précision)
    )

    # Valeurs des paramètres à l'optimum
    _, obj_values, lca_parameters = drone_sizing(performances, technologies, param_acv, lcia_methods_dict, variables_conception_opt)
    sizing_df = pd.DataFrame.from_dict({'Drone mass [kg]': obj_values[0], 
                              'Propellers diameter [m]': variables_conception_opt[0], 
                              'Motors nominal torque [N.m]': variables_conception_opt[1],
                              'Battery energy [Wh]': variables_conception_opt[2],
                              'Arms length [m]': variables_conception_opt[3]
                             }, orient='index')
    lca_df = pd.DataFrame.from_dict(lca_parameters, orient='index', columns=['value'])
    
    return sizing_df, lca_df
