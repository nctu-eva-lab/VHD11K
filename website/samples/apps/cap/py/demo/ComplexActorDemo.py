import time
from termcolor import colored
from autogencap.LocalActorNetwork import LocalActorNetwork
from AppAgents import FidelityAgent, FinancialPlannerAgent, PersonalAssistant, QuantAgent, RiskManager


def complex_actor_demo():
    """
    This function demonstrates the usage of a complex actor system.
    It creates a local actor graph, registers various agents,
    connects them, and interacts with a personal assistant agent.
    The function continuously prompts the user for input messages,
    sends them to the personal assistant agent, and terminates
    when the user enters "quit".
    """
    network = LocalActorNetwork()
    # Register agents
    network.register(PersonalAssistant())
    network.register(FidelityAgent())
    network.register(FinancialPlannerAgent())
    network.register(RiskManager())
    network.register(QuantAgent())
    # Tell agents to connect to other agents
    network.connect()
    # Get a channel to the personal assistant agent
    pa = network.lookup_actor(PersonalAssistant.cls_agent_name)
    info_msg = """
    This is an imaginary personal assistant agent scenario.
    Five actors are connected in a self-determined graph. The user
    can interact with the personal assistant agent by entering
    their name. The personal assistant agent will then enlist
    the other four agents to create a financial plan.

    Start by entering your name.
    """
    print(colored(info_msg, "blue"))

    while True:
        # For aesthetic reasons, better to let network messages complete
        time.sleep(0.1)
        # Get a message from the user
        msg = input(colored("Enter your name (or quit): ", "light_red"))
        # Send the message to the personal assistant agent
        pa.send_txt_msg(msg)
        if msg.lower() == "quit":
            break
    # Cleanup

    pa.close()
    network.disconnect()
