from entities.scene import Scene
from entities.vector2d import Point2D
from entities.frame_object import PersonInFrame, FrameObject
from entities.scene_features import DatapointFeatures, SceneFeatures
from tqdm import tqdm

class FeatureExtractor:
    INDIVIDUAL_FTS_DIM = 8
    INTERACTION_FTS_DIM = (None, 5)

    def __init__(self, print_progress: bool = True):
        self.print_progress = print_progress
            
    def extract_all_scenes_features(
        self,
        scenes: dict[int, Scene]
    ) -> dict[int, SceneFeatures]: 
        scenes_features = {}
        for scene_id, scene in tqdm(
            scenes.items(), 
            desc=f"[dataset_name={getattr(scene, 'dataset')}] Extracting features from scenes", 
            disable=not self.print_progress
            ):
            scenes_features[scene_id] = self.extract_scene_features(scene)
        return scenes_features

    def extract_scene_features(self, scene: Scene) -> SceneFeatures:
        frames_features = []
        for frame in scene.frames:
            frame_features = self.extract_frame_features(
                scene.focus_person_ids, 
                frame.frame_objects, 
                scene.goal_positions
            )
            frames_features.update(frame_features)
        return frames_features

    def extract_frame_features(
        self,
        focus_person_ids: list[int], 
        frame_objs: list[FrameObject], 
        goal_positions: dict[int, Point2D]
    ) -> list[DatapointFeatures]:
        frame_features = []
        for focus_person_id in focus_person_ids:
            focus_person_in_frame = next(
                (o for o in frame_objs if isinstance(o, PersonInFrame) and o.id == focus_person_id), None
            )
            
            if focus_person_in_frame is None:
                continue
            
            individual_features = self.get_individual_features(
                focus_person_in_frame, 
                goal_positions[focus_person_id] 
            )
            interaction_features = self.get_interaction_features(focus_person_in_frame, frame_objs)
            focus_person_features = (
                DatapointFeatures(
                    individual_features=individual_features,
                    interaction_features=interaction_features
                )
            )
            frame_features.append(focus_person_features)
        return frame_features

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