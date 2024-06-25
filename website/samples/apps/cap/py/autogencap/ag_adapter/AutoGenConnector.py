from typing import Dict, Optional, Union
from autogen import Agent
from ..ActorConnector import ActorConnector
from ..proto.Autogen_pb2 import GenReplyReq, GenReplyResp, PrepChat, ReceiveReq, Terminate


class AutoGenConnector:
    """
    A specialized ActorConnector class for sending and receiving Autogen messages
    to/from the CAP system.
    """

    def __init__(self, cap_sender: ActorConnector):
        self._can_channel: ActorConnector = cap_sender

    def close(self):
        """
        Close the connector.
        """
        self._can_channel.close()

    def _send_msg(self, msg):
        """
        Send a message to CAP.
        """
        serialized_msg = msg.SerializeToString()
        self._can_channel.send_bin_msg(type(msg).__name__, serialized_msg)

    def send_gen_reply_req(self):
        """
        Send a GenReplyReq message to CAP and receive the response.
        """
        msg = GenReplyReq()
        serialized_msg = msg.SerializeToString()
        _, _, _, resp = self._can_channel.binary_request(type(msg).__name__, serialized_msg)
        gen_reply_resp = GenReplyResp()
        gen_reply_resp.ParseFromString(resp)
        return gen_reply_resp.data

    def send_receive_req(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """
        Send a ReceiveReq message to CAP.
        """
        msg = ReceiveReq()
        if isinstance(message, dict):
            for key, value in message.items():
                msg.data_map.data[key] = value
        elif isinstance(message, str):
            msg.data = message
        msg.sender = sender.name
        if request_reply is not None:
            msg.request_reply = request_reply
        if silent is not None:
            msg.silent = silent
        self._send_msg(msg)

    def send_terminate(self, sender: Agent):
        """
        Send a Terminate message to CAP.
        """
        msg = Terminate()
        msg.sender = sender.name
        self._send_msg(msg)

    def send_prep_chat(self, recipient: "Agent", clear_history: bool, prepare_recipient: bool = True) -> None:
        """
        Send a PrepChat message to CAP.

        Args:
            recipient (Agent): _description_
            clear_history (bool): _description_
            prepare_recipient (bool, optional): _description_. Defaults to True.
        """
        msg = PrepChat()
        msg.recipient = recipient.name
        msg.clear_history = clear_history
        msg.prepare_recipient = prepare_recipient
        self._send_msg(msg)
