import zmq
from autogencap.Actor import Actor
from autogencap.Constants import Termination_Topic
from autogencap.DebugLog import Debug


class AGActor(Actor):
    def start(self, context: zmq.Context):
        super().start(context)
        str_topic = Termination_Topic
        Debug(self.actor_name, f"subscribe to: {str_topic}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, f"{str_topic}")
