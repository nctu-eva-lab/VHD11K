import time
import _paths
from autogen import UserProxyAgent, config_list_from_json
from autogencap.DebugLog import Info
from autogencap.LocalActorNetwork import LocalActorNetwork
from autogencap.ag_adapter.CAP2AG import CAP2AG
from autogencap.Config import IGNORED_LOG_CONTEXTS


# Starts the Broker and the Assistant. The UserProxy is started separately.
class StandaloneUserProxy:
    def __init__(self):
        pass

    def run(self):
        print("Running the StandaloneUserProxy")

        user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": "coding"},
            is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
        )
        # Composable Agent Network adapter
        network = LocalActorNetwork()
        user_proxy_adptr = CAP2AG(ag_agent=user_proxy, the_other_name="assistant", init_chat=True, self_recursive=True)
        network.register(user_proxy_adptr)
        network.connect()

        # Send a message to the user_proxy
        user_proxy_conn = network.lookup_actor("user_proxy")
        example = "Plot a chart of MSFT daily closing prices for last 1 Month."
        print(f"Example: {example}")
        try:
            user_input = input("Please enter your command: ")
            if user_input == "":
                user_input = example
            print(f"Sending: {user_input}")
            user_proxy_conn.send_txt_msg(user_input)

            # Hang around for a while
            while user_proxy_adptr.run:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down.")
        network.disconnect()
        Info("StandaloneUserProxy", "App Exit")


def main():
    IGNORED_LOG_CONTEXTS.extend(["BROKER", "DirectorySvc"])
    assistant = StandaloneUserProxy()
    assistant.run()


if __name__ == "__main__":
    main()
