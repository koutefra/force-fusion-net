from entities.scene import Scene
from entities.vector2d import Point2D
from entities.frame_object import PersonInFrame, FrameObject

class FeatureExtractor:
    PERSON_FEATURES_DIM = 8
    INTERACTION_FEATURES_DIM = (None, 5)
            
    @staticmethod
    def extract_all_scenes_features(scenes: dict[int, Scene]) -> dict[int, list[tuple[dict[str, float], list[dict[str, float]]]]]:
        scenes_features = {}
        for scene_id, scene in scenes.items():
            scenes_features[scene_id] = FeatureExtractor.extract_scene_features(scene)
        return scenes_features

    @staticmethod
    def extract_scene_features(scene: Scene) -> list[tuple[dict[str, float], list[dict[str, float]]]]:
        frames_features = []
        person_id = scene.focus_person_id
        goal_pos = scene.focus_person_goal
        for frame in scene.frames:
            frame_features = FeatureExtractor.extract_frame_features(
                person_id, 
                frame.frame_objects, 
                goal_pos
            )
            frames_features.append(frame_features)
        return frames_features

    @staticmethod
    def extract_frame_features(person_id: int, frame_objs: list[FrameObject], goal_pos: Point2D) -> tuple[dict[str, float], list[dict[str, float]]]:
        person = next((obj for obj in frame_objs if isinstance(obj, PersonInFrame) and obj.id == person_id), None)
        
        if person is None:
            raise ValueError(f"Person with ID {person_id} not found in frame objects.")
        
        individual_features = FeatureExtractor.get_individual_features(
            person, 
            goal_pos
        )
        interaction_features = FeatureExtractor.get_interaction_features(person, frame_objs)
        
        return individual_features, interaction_features

    @staticmethod
    def get_individual_features(person: PersonInFrame, goal_pos: Point2D) -> dict[str, float]:
        dist_to_goal = (person.position - goal_pos).magnitude()
        dir_to_goal = person.position.direction_to(goal_pos)
        dir_to_goal_norm = dir_to_goal.normalize()
        vel_towards_goal = person.velocity.dot(dir_to_goal_norm)
        return {
            "position_x": person.position.x,
            "position_y": person.position.y,
            "velocity_x": person.velocity.x,
            "velocity_y": person.velocity.y,
            "distance_to_goal": dist_to_goal,
            "direction_x_to_goal": dir_to_goal.x,
            "direction_y_to_goal": dir_to_goal.y,
            "velocity_towards_goal": vel_towards_goal,
            "acceleration_x": person.acceleration.x,
            "acceleration_y": person.acceleration.y
        }

    @staticmethod
    def get_interaction_features(person: PersonInFrame, frame_objs: list[FrameObject]) -> list[dict[str, float]]:
        interaction_features = []
        for frame_obj in frame_objs:
            if not isinstance(frame_obj, PersonInFrame):
                continue
            other_person = frame_obj
            if person.id == other_person.id:
                continue
            
            distance = (person.position - other_person.position).magnitude()
            direction_vector = person.position.direction_to(other_person.position)
            relative_velocity = (person.velocity - other_person.velocity).magnitude()
            alignment = person.velocity.dot(direction_vector)
            
            interaction_features.append({
                f"distance_to_person_{other_person.id}": distance,
                f"direction_x_to_person_{other_person.id}": direction_vector.x,
                f"direction_y_to_person_{other_person.id}": direction_vector.y,
                f"relative_velocity_to_person_{other_person.id}": relative_velocity,
                f"alignment_to_person_{other_person.id}": alignment,
            })
        return interaction_features