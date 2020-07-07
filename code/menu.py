from consolemenu import ConsoleMenu
from consolemenu import SelectionMenu
from consolemenu.items import FunctionItem, SubmenuItem, CommandItem

import util
from bayes_nets import JunctionTree

models_path = "models/"
available_models = {'Cancer scan': 'cancer.dat', 'Chest clinic': 'chestclinic.dat', 'Fire alarm': 'fire.dat',
                    'Monty python': 'monty.dat', 'Poker': 'poker.dat', 'Horse farm': 'studfarm.dat'}
models_list = list(available_models.keys())


def show_first_menu():
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

    net, jtree = util.load_model(models_path + available_models[selected_model])
    if net is not None and jtree is not None:
        jtree.initialize_tables(net)
        print("Model loaded succesfully")
        show_loaded_model_menu(selected_model, net, jtree)
    else:
        show_first_menu()


def show_loaded_model_menu(model, net, jtree):
    menu = ConsoleMenu("Main menu - " + model)

    describe_bnet = FunctionItem("Describe Bayesian Net", function= lambda net : print(str(net)), args=[net])
    describe_jtree = FunctionItem("Describe Junction Tree", function=lambda jtree: print(str(jtree)), args=[jtree])

    propagate = FunctionItem("Propagate evidence", function=lambda jtree: jtree.sum_propagate(), args=[jtree])

    load_new_model = SelectionMenu(models_list, "Load New Model")
    load_new_model_item = SubmenuItem("Load a new model", load_new_model, menu=menu, should_exit=True)

    add_evidence = FunctionItem("Add evidence", function=add_evidence_option, args=[jtree])
    get_probability = FunctionItem("Get probability of a variable", function=get_probability_option, args=[jtree])
    reset_model = FunctionItem("Reset Model", function=lambda net,jtree : jtree.initialize_tables(net), args=[net,jtree])

    menu.append_item(describe_bnet)
    menu.append_item(describe_jtree)
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
    net, jtree = util.load_model(models_path + available_models[selected_model])
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
    (chosen_var, chosen_value) = in_str.split(',')
    if chosen_var is None or chosen_value is None:
        print("Input error")

    jtree.add_evidence(chosen_var, chosen_value)


def get_probability_option(jtree: JunctionTree):
    vars = list(jtree.get_variables().keys())
    for var in vars:
        print(var)

    in_str = input("Insert the name of the variable you want to see the probabilities of:").strip()

    print(jtree.calculate_variable_probability(in_str))
