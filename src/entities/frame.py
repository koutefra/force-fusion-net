from dataclasses import dataclass
from entities.frame_object import FrameObject

@dataclass(frozen=True)
class Frame:
    number: int
    frame_objects: list[FrameObject]