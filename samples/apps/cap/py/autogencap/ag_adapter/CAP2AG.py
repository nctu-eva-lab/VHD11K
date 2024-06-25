from enum import Enum
from typing import Optional
from ..DebugLog import Debug, Error, Info, Warn, shorten
from ..LocalActorNetwork import LocalActorNetwork
from ..proto.Autogen_pb2 import GenReplyReq, GenReplyResp, PrepChat, ReceiveReq, Terminate
from .AGActor import AGActor
from .AG2CAP import AG2CAP
from autogen import ConversableAgent


class CAP2AG(AGActor):
    """
    A CAN actor that acts as an adapter for the AutoGen system.
    """

    States = Enum("States", ["INIT", "CONVERSING"])

    def __init__(self, ag_agent: ConversableAgent, the_other_name: str, init_chat: bool, self_recursive: bool = True):
        super().__init__(ag_agent.name, ag_agent.description)
        self._the_ag_agent: ConversableAgent = ag_agent
        self._ag2can_other_agent: AG2CAP = None
        self._other_agent_name: str = the_other_name
        self._init_chat: bool = init_chat
        self.STATE = self.States.INIT
        self._can2ag_name: str = self.actor_name + ".can2ag"
        self._self_recursive: bool = self_recursive
        self._network: LocalActorNetwork = None
        self._connectors = {}

    def connect_network(self, network: LocalActorNetwork):
        """
        Connect to the AutoGen system.
        """
        self._network = network
        self._ag2can_other_agent = AG2CAP(self._network, self._other_agent_name)
        Debug(self._can2ag_name, "connected to {network}")

    def disconnect_network(self, network: LocalActorNetwork):
        """
        Disconnect from the AutoGen system.
        """
        super().disconnect_network(network)
        #        self._the_other.close()
        Debug(self.actor_name, "disconnected")

    def _process_txt_msg(self, msg: str, msg_type: str, topic: str, sender: str):
        """
        Process a text message received from the AutoGen system.
        """
        Info(self._can2ag_name, f"proc_txt_msg: [{topic}], [{msg_type}], {shorten(msg)}")
        if self.STATE == self.States.INIT:
            self.STATE = self.States.CONVERSING
            if self._init_chat:
                self._the_ag_agent.initiate_chat(self._ag2can_other_agent, message=msg, summary_method=None)
            else:
                self._the_ag_agent.receive(msg, self._ag2can_other_agent, True)
        else:
            self._the_ag_agent.receive(msg, self._ag2can_other_agent, True)
        return True

    def _call_agent_receive(self, receive_params: ReceiveReq):
        request_reply: Optional[bool] = None
        silent: Optional[bool] = False

        if receive_params.HasField("request_reply"):
            request_reply = receive_params.request_reply
        if receive_params.HasField("silent"):
            silent = receive_params.silent

        save_name = self._ag2can_other_agent.name
        self._ag2can_other_agent.set_name(receive_params.sender)
        if receive_params.HasField("data_map"):
            data = dict(receive_params.data_map.data)
        else:
            data = receive_params.data
        self._the_ag_agent.receive(data, self._ag2can_other_agent, request_reply, silent)
        self._ag2can_other_agent.set_name(save_name)

    def receive_msgproc(self, msg: bytes):
        """
        Process a ReceiveReq message received from the AutoGen system.
        """
        receive_params = ReceiveReq()
        receive_params.ParseFromString(msg)

        self._ag2can_other_agent.reset_receive_called()

        if self.STATE == self.States.INIT:
            self.STATE = self.States.CONVERSING

            if self._init_chat:
                self._the_ag_agent.initiate_chat(
                    self._ag2can_other_agent, message=receive_params.data, summary_method=None
                )
            else:
                self._call_agent_receive(receive_params)
        else:
            self._call_agent_receive(receive_params)

        if not self._ag2can_other_agent.was_receive_called() and self._self_recursive:
            Warn(self._can2ag_name, "TERMINATE")
            self._ag2can_other_agent.send_terminate(self._the_ag_agent)
            return False
        return True

    def get_actor_connector(self, topic: str):
        """
        Get the actor connector for the given topic.
        """
        if topic in self._connectors:
            return self._connectors[topic]
        else:
            connector = self._network.actor_connector_by_topic(topic)
            self._connectors[topic] = connector
            return connector

    def generate_reply_msgproc(self, msg: GenReplyReq, sender_topic: str):
        """
        Process a GenReplyReq message received from the AutoGen system and generate a reply.
        """
        generate_reply_params = GenReplyReq()
        generate_reply_params.ParseFromString(msg)
        reply = self._the_ag_agent.generate_reply(sender=self._ag2can_other_agent)
        connector = self.get_actor_connector(sender_topic)

        reply_msg = GenReplyResp()
        if reply:
            reply_msg.data = reply.encode("utf8")
        serialized_msg = reply_msg.SerializeToString()
        connector.send_bin_msg(type(reply_msg).__name__, serialized_msg)
        return True

    def prepchat_msgproc(self, msg, sender_topic):
        prep_chat = PrepChat()
        prep_chat.ParseFromString(msg)
        self._the_ag_agent._prepare_chat(self._ag2can_other_agent, prep_chat.clear_history, prep_chat.prepare_recipient)
        return True

    def _process_bin_msg(self, msg: bytes, msg_type: str, topic: str, sender: str):
        """
        Process a binary message received from the AutoGen system.
        """
        Info(self._can2ag_name, f"proc_bin_msg: topic=[{topic}], msg_type=[{msg_type}]")
        if msg_type == ReceiveReq.__name__:
            return self.receive_msgproc(msg)
        elif msg_type == GenReplyReq.__name__:
            return self.generate_reply_msgproc(msg, sender)
        elif msg_type == PrepChat.__name__:
            return self.prepchat_msgproc(msg, sender)
        elif msg_type == Terminate.__name__:
            Warn(self._can2ag_name, f"TERMINATE received: topic=[{topic}], msg_type=[{msg_type}]")
            return False
        else:
            Error(self._can2ag_name, f"Unhandled message type: topic=[{topic}], msg_type=[{msg_type}]")
        return True
