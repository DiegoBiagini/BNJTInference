import os
from os import listdir
from os.path import isfile, join

from consolemenu import ConsoleMenu
from consolemenu import SelectionMenu
from consolemenu.items import FunctionItem, SubmenuItem

import util
from bayes_nets import JunctionTree

models_path = "models/"


def show_first_menu():
    # Record which models are in the models folder
    models_list = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f[-4:] == ".dat"]

    first_menu = ConsoleMenu("Main menu")

    submenu = SelectionMenu(models_list, "Load Model")
    submenu_item = SubmenuItem("Load a model", submenu, menu=first_menu, should_exit=True)

    first_menu.append_item(submenu_item)

    first_menu.start()
    first_menu.join()

    if submenu.selected_option >= len(models_list):
        show_first_menu()
        return
    elif submenu.selected_option == -1:
        return

    selected_model = models_list[submenu.selected_option]

    net, jtree = util.load_model(models_path + selected_model)
    if net is not None and jtree is not None:
        jtree.initialize_tables(net)
        print("Model loaded succesfully")
        show_loaded_model_menu(selected_model, net, jtree)
    else:
        show_first_menu()


def show_loaded_model_menu(model, net, jtree):
    # Record which models are in the models folder
    models_list = [f for f in listdir(models_path) if isfile(join(models_path, f)) and str(f)[-4:] == ".dat"]

    menu = ConsoleMenu("Main menu - " + model)

    describe_bnet = FunctionItem("Describe Bayesian Net", function= lambda net : print(str(net)), args=[net])

    describe_jtree = FunctionItem("Describe Junction Tree", function=lambda jtree: print(str(jtree)), args=[jtree])

    visualize = FunctionItem("Visualize Bayesian Net and Junction Tree", function=lambda path: os.system("visualize.py " + path), args=[models_path + model])

    propagate = FunctionItem("Propagate evidence", function=lambda jtree: [jtree.sum_propagate(), print("Evidence propagated")], args=[jtree])

    load_new_model = SelectionMenu(models_list, "Load New Model")
    load_new_model_item = SubmenuItem("Load a new model", load_new_model, menu=menu, should_exit=True)

    add_evidence = FunctionItem("Add evidence", function=add_evidence_option, args=[jtree])
    get_probability = FunctionItem("Get probability of a variable", function=get_probability_option, args=[jtree])
    reset_model = FunctionItem("Reset Model", function=lambda net, jtree : [jtree.initialize_tables(net), print("Tables reinitialized")], args=[net, jtree])

    menu.append_item(describe_bnet)
    menu.append_item(describe_jtree)
    menu.append_item(visualize)
    menu.append_item(add_evidence)
    menu.append_item(get_probability)
    menu.append_item(propagate)
    menu.append_item(reset_model)
    menu.append_item(load_new_model_item)

    menu.start()
    menu.join()

    if load_new_model.selected_option >= len(models_list):
        show_loaded_model_menu(model, net, jtree)
        return
    elif load_new_model.selected_option == -1:
        return

    selected_model = models_list[load_new_model.selected_option]
    net, jtree = util.load_model(models_path + selected_model)
    if net is not None and jtree is not None:
        jtree.initialize_tables(net)
        print("Model loaded succesfully")
        show_loaded_model_menu(selected_model, net, jtree)
    else:
        show_first_menu()


def add_evidence_option(jtree):
    vars = list(jtree.get_variables().keys())
    for var in vars:
        print(var)

    in_str = input("Insert evidence in the form <variable>,<value> :").strip()
    try:
        (chosen_var, chosen_value) = in_str.split(',')
        if chosen_var is None or chosen_value is None:
            print("Input error")

        jtree.add_evidence(chosen_var, chosen_value)
        print("Evidence entered successfully")
    except ValueError:
        print("Wrong format")
    except RuntimeError:
        print("Conflicting evidence was entered, operation aborted")


def get_probability_option(jtree: JunctionTree):
    vars = list(jtree.get_variables().keys())
    for var in vars:
        print(var)

    in_str = input("Insert the name of the variable you want to see the probabilities of:").strip()

    print(jtree.calculate_variable_probability(in_str))

