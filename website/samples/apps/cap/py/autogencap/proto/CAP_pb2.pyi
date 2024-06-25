from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class ActorInfo(_message.Message):
    __slots__ = ("name", "namespace", "description")
    NAME_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    namespace: str
    description: str
    def __init__(
        self, name: _Optional[str] = ..., namespace: _Optional[str] = ..., description: _Optional[str] = ...
    ) -> None: ...

class ActorRegistration(_message.Message):
    __slots__ = ("actor_info",)
    ACTOR_INFO_FIELD_NUMBER: _ClassVar[int]
    actor_info: ActorInfo
    def __init__(self, actor_info: _Optional[_Union[ActorInfo, _Mapping]] = ...) -> None: ...

class ActorLookup(_message.Message):
    __slots__ = ("actor_info",)
    ACTOR_INFO_FIELD_NUMBER: _ClassVar[int]
    actor_info: ActorInfo
    def __init__(self, actor_info: _Optional[_Union[ActorInfo, _Mapping]] = ...) -> None: ...

class ActorInfoCollection(_message.Message):
    __slots__ = ("info_coll",)
    INFO_COLL_FIELD_NUMBER: _ClassVar[int]
    info_coll: _containers.RepeatedCompositeFieldContainer[ActorInfo]
    def __init__(self, info_coll: _Optional[_Iterable[_Union[ActorInfo, _Mapping]]] = ...) -> None: ...

class ActorLookupResponse(_message.Message):
    __slots__ = ("found", "actor")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    found: bool
    actor: ActorInfoCollection
    def __init__(self, found: bool = ..., actor: _Optional[_Union[ActorInfoCollection, _Mapping]] = ...) -> None: ...

class Ping(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Pong(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
