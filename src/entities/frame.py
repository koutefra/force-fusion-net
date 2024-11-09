from dataclasses import dataclass
from entities.frame_object import FrameObject

@dataclass(frozen=True)
class Frame:
    number: int
    frame_objects: list[FrameObject]

    def transform_object(self, transformed_object: FrameObject) -> "Frame":
        new_frame_objects = []
        for o in self.frame_objects:
            if type(o) == type(transformed_object) and o.id == transformed_object.id:
                new_frame_objects.append(transformed_object)
            else:
                new_frame_objects.append(o)
        return Frame(number=self.number, frame_objects=new_frame_objects)
        