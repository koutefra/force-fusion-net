import numpy as np
from entities.frame import Frame, Frames
from entities.vector2d import closest_point_on_line
from entities.vector2d import Point2D
from entities.obstacle import LineObstacle

class FrameEvaluator:
    @staticmethod
    def center_of_mass(frame: Frame) -> Point2D:
        """
        Compute the center of mass of all people in the frame.
        """
        if not frame.persons:
            return Point2D(0.0, 0.0)
        
        positions = np.array([[person.position.x, person.position.y] for person in frame.persons.values()])
        center = np.mean(positions, axis=0) if len(positions) > 0 else [0.0, 0.0]
        
        return Point2D(center[0], center[1])

    @staticmethod
    def count_agent_collisions(frame: Frame, threshold: float) -> int:
        """
        Count the number of collisions between agents in the frame.
        Collision is detected if two agents are closer than 'threshold'.
        """
        collisions = 0
        people = list(frame.persons.values())
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                dist = np.linalg.norm(people[i].position - people[j].position)
                if dist < threshold:
                    collisions += 1
        return collisions

    @staticmethod
    def count_obstacle_collisions(frame: Frame, threshold: float) -> int:
        """
        Count the number of agents colliding with obstacles.
        Obstacles are represented as points or bounding box centers.
        """
        collisions = 0
        for person in frame.persons.values():
            for obs in frame.obstacles:
                p_closest = closest_point_on_line(person.position, obs.p1, obs.p2)
                dist = np.linalg.norm(person.position - p_closest)
                if dist < threshold:
                    collisions += 1
        return collisions

    def evaluate_frame(self, frame: Frame, agent_coll_thr: float, obstacle_coll_thr: float) -> dict[str, float]:
        """
        Compare two frames and compute various metrics.
        """
        center_of_mass = self.center_of_mass(frame)
        agent_collisions_sim = self.count_agent_collisions(frame, threshold=agent_coll_thr)
        obstacle_collisions_sim = self.count_obstacle_collisions(frame, threshold=obstacle_coll_thr)
        velocity_magnitudes = np.array([p.velocity.magnitude() for p in frame.persons.values() if p.velocity])
        return {
            "center_of_mass": center_of_mass,
            "agent_collisions": agent_collisions_sim,
            "obstacle_collisions": obstacle_collisions_sim,
            "velocity_avg": np.mean(velocity_magnitudes),
            "velocity_max": np.max(velocity_magnitudes),
            "velocity_std": np.std(velocity_magnitudes),
        }

    def evaluate_frames(
        self, 
        frames: Frames,
        agent_coll_thr: float,
        obstacle_coll_thr: float,
    ) -> dict[str, float]:
        agent_collisions = 0
        obstacle_collisions = 0
        center_of_mass_x_sum, center_of_mass_y_sum = 0.0, 0.0
        velocity_magnitudes = []
        for frame in frames.values():
            agent_collisions += self.count_agent_collisions(frame, agent_coll_thr)
            obstacle_collisions += self.count_obstacle_collisions(frame, obstacle_coll_thr)
            center_of_mass = self.center_of_mass(frame)
            center_of_mass_x_sum += center_of_mass.x
            center_of_mass_y_sum += center_of_mass.y
            velocity_magnitudes.extend([p.velocity.magnitude() for p in frame.persons.values() if p.velocity])
        velocity_magnitudes = np.array(velocity_magnitudes) if velocity_magnitudes else np.array([0.0])
        return {
            "agent_collisions": agent_collisions,
            "obstacle_collisions": obstacle_collisions,
            "center_of_mass_x_avg": center_of_mass_x_sum / len(frames),
            "center_of_mass_y_avg": center_of_mass_y_sum / len(frames),
            "velocity_avg": np.mean(velocity_magnitudes),
            "velocity_max": np.max(velocity_magnitudes),
            "velocity_std": np.std(velocity_magnitudes),
        }