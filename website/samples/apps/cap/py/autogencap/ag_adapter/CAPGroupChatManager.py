from autogen import GroupChatManager
from autogencap.ActorConnector import ActorConnector
from autogencap.LocalActorNetwork import LocalActorNetwork
from autogencap.ag_adapter.CAP2AG import CAP2AG
from autogencap.ag_adapter.CAPGroupChat import CAPGroupChat
import time


class CAPGroupChatManager:
    def __init__(self, groupchat: CAPGroupChat, llm_config: dict, network: LocalActorNetwork):
        self._network: LocalActorNetwork = network
        self._cap_group_chat: CAPGroupChat = groupchat
        self._ag_group_chat_manager: GroupChatManager = GroupChatManager(
            groupchat=self._cap_group_chat, llm_config=llm_config
        )
        self._cap_proxy: CAP2AG = CAP2AG(
            ag_agent=self._ag_group_chat_manager,
            the_other_name=self._cap_group_chat.chat_initiator,
            init_chat=False,
            self_recursive=True,
        )
        self._network.register(self._cap_proxy)

    def initiate_chat(self, txt_msg: str) -> None:
        self._network.connect()
        user_proxy_conn: ActorConnector = self._network.lookup_actor(self._cap_group_chat.chat_initiator)
        user_proxy_conn.send_txt_msg(txt_msg)
        self._wait_for_user_exit()

    def is_running(self) -> bool:
        return self._cap_group_chat.is_running()

    def _wait_for_user_exit(self) -> None:
        try:
            while self.is_running():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down.")
