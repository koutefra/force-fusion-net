import torch
import json
import itertools
from tqdm import tqdm
import torch.nn as nn
from models.base_model import BaseModel
from entities.batched_frames import BatchedFrames
from entities.scene import Scene
import math

class SocialForceB160(BaseModel):
    return_positions_in_features = True
    ALWAYS_ENFORCE_OUTER_WALLS = True  # set to False to use strictly (a)-(d) only

    # default values taken from: https://pedestriandynamics.org/models/social_force_model
    def __init__(
        self, 
        A_interaction: float = 50.0,  # Interaction force constant, (m/s^(-2)) 
        A_obstacle: float = 50.0,     # Interaction force constant, (m/s^(-2)) 
        B_interaction: float = 0.08,  # Interaction decay constant, m
        B_obstacle: float = 0.08,     # Interaction decay constant, m
        tau: float = 0.5,             # Relaxation time constant, s
        desired_speed: float = 0.8,   # m/s
    ):
        super(SocialForceB160, self).__init__()
        self.A_interaction = nn.Parameter(torch.tensor(A_interaction))
        self.A_obstacle = nn.Parameter(torch.tensor(A_obstacle))
        self.B_interaction = nn.Parameter(torch.tensor(B_interaction))
        self.B_obstacle = nn.Parameter(torch.tensor(B_obstacle))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.desired_speed = torch.tensor(desired_speed)

    def forward_single(
        self, 
        x_individual: torch.Tensor, 
        interaction_features: tuple[torch.Tensor, torch.Tensor],
        obstacle_features: tuple[torch.Tensor, torch.Tensor],  # unused here (we hardcode b160)
    ) -> torch.Tensor:
        x_interaction, interaction_mask = interaction_features

        fmap = BatchedFrames.get_individual_feature_mapping(return_positions=True)
        pos_x = x_individual[:, fmap['pos_x']]
        pos_y = x_individual[:, fmap['pos_y']]
        vel_x = x_individual[:, fmap['vel_x']]
        vel_y = x_individual[:, fmap['vel_y']]

        # --- Desired force with (0,0) gate when x<0 and |y|>=0.8 ---
        use_gate = (pos_x < 0.0) & ((pos_y <= -0.8) | (pos_y >= 0.8))

        dir_goal_x = x_individual[:, fmap['goal_dir_x']]
        dir_goal_y = x_individual[:, fmap['goal_dir_y']]

        gate_vec_x = -pos_x
        gate_vec_y = -pos_y
        gate_norm = torch.clamp(torch.sqrt(gate_vec_x**2 + gate_vec_y**2), min=1e-9)
        gate_dir_x = gate_vec_x / gate_norm
        gate_dir_y = gate_vec_y / gate_norm

        dir_x = torch.where(use_gate, gate_dir_x, dir_goal_x)
        dir_y = torch.where(use_gate, gate_dir_y, dir_goal_y)

        desired_force = self._desired_force(dir_x, dir_y, vel_x, vel_y)

        # --- Interaction force (unchanged) ---
        interaction_force = self._compute_force(
            x_interaction[:, :, BatchedFrames.get_interaction_feature_index('dir_x')],
            x_interaction[:, :, BatchedFrames.get_interaction_feature_index('dir_y')],
            x_interaction[:, :, BatchedFrames.get_interaction_feature_index('dist')],
            self.A_interaction, self.B_interaction, interaction_mask
        )

        # --- Analytic obstacle force for b160 (loop-based, (y,x)->(x,y) conversion) ---
        positions = torch.stack((pos_x, pos_y), dim=-1)  # [n,2] as (x,y)
        obstacle_force = self._compute_obstacle_force_b160(positions)

        total_force = desired_force + interaction_force + obstacle_force
        return total_force

    def _desired_force(
        self, 
        dir_x_to_goal: torch.Tensor, 
        dir_y_to_goal: torch.Tensor,
        velocity_x: torch.Tensor,
        velocity_y: torch.Tensor
    ) -> torch.Tensor:
        dir_to_goal = torch.stack((dir_x_to_goal, dir_y_to_goal), dim=-1)
        norm = torch.clamp(torch.norm(dir_to_goal, dim=-1, keepdim=True), min=1e-9)
        normalized_dir_to_goal = dir_to_goal / norm
        desired_velocity = normalized_dir_to_goal * self.desired_speed
        velocity = torch.stack((velocity_x, velocity_y), dim=-1)
        desired_acceleration = (desired_velocity - velocity) / self.tau
        return desired_acceleration  # [n,2]

    # ---------- NEW: loop-based b160 obstacle computation ----------
    def _compute_obstacle_force_b160(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Hardcoded obstacles per your b160 rules.
        IMPORTANT: Your endpoints are specified as (y, x). We convert them to (x, y).
        Selection precedence:
          (a) y < -0.8  -> two 'wall' segments
          (b) y >  0.8  -> two 'wall' segments (top-side counterpart)
          (c) x <  0    -> two 'point' obstacles at y=±0.8 on x=0
          (d) x >= 0    -> two 'line' obstacles along x=0 from y=±0.8 to 4
        Additionally (default): always enforce bottom outer walls at x=±3.5 for agents with y<0,
        so agents starting near the lower corners cannot leave through those walls.
        """
        device = pos.device
        out = torch.zeros_like(pos)  # [n,2]

        # helpers (scalar math; no vectorization)
        def closest_point_on_segment(px, py, ax, ay, bx, by):
            abx, aby = bx - ax, by - ay
            apx, apy = px - ax, py - ay
            ab2 = abx*abx + aby*aby
            if ab2 <= 1e-12:
                return ax, ay
            t = (apx*abx + apy*aby) / ab2
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
            cx = ax + t * abx
            cy = ay + t * aby
            return cx, cy

        def add_segment_force(px, py, ax, ay, bx, by):
            cx, cy = closest_point_on_segment(px, py, ax, ay, bx, by)
            dx, dy = cx - px, cy - py
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                return 0.0, 0.0
            dirx, diry = dx / dist, dy / dist
            mag = float(self.A_obstacle.item()) * math.exp(-dist / float(self.B_obstacle.item()))
            # repulsion is opposite to direction from agent->closest point
            return -dirx * mag, -diry * mag

        def add_point_force(px, py, qx, qy):
            dx, dy = qx - px, qy - py
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                return 0.0, 0.0
            dirx, diry = dx / dist, dy / dist
            mag = float(self.A_obstacle.item()) * math.exp(-dist / float(self.B_obstacle.item()))
            return -dirx * mag, -diry * mag

        n = pos.size(0)
        for i in range(n):
            px = float(pos[i, 0].item())
            py = float(pos[i, 1].item())
            fx, fy = 0.0, 0.0

            # (a) pos_y < -0.8
            if py < -0.8:
                a1 = (-3.5, -3.5); b1 = (0.0, -3.5)
                a2 = (0.0,  -3.5); b2 = (0.0, -0.8)  # (x,y): ( 0.0,-3.5) -> (0.0,-0.8)  vertical at x=0
                segs = [ (a1, b1), (a2, b2) ]

            # (b) pos_y > 0.8
            elif py > 0.8:
                a1 = (-3.5, 3.5); b1 = (0.0, 3.5)  # (x,y): (-3.5,3.5) -> (0.0,3.5)  horizontal top-left
                a2 = (0.0, 3.5); b2 = (0.0, 0.8)  # (x,y): ( 0.0,3.5) -> (0.0,0.8)  vertical at x=0
                segs = [ (a1, b1), (a2, b2) ]

            # (c) center band and x < 0  -> two points (y,x): (-0.8,0.0) and (0.8,0.0)
            elif px < 0.0:
                points = [ (0.0, -0.8), (0.0, 0.8) ]  # (x,y): (0.0,-0.8) and (0.0,0.8)
                segs = []
            # (d) center band and x >= 0  -> two vertical lines along x=0 from y=±0.8 to 4
            else:
                a1 = (0.0, -0.8); b1 = (4.0, -0.8)   # (x,y): (0.0,-0.8) -> (4.0,-0.8)
                a2 = (0.0, 0.8); b2 = (4.0, 0.8)   # (x,y): (0.0, 0.8) -> (4.0,0.8)
                segs = [ (a1, b1), (a2, b2) ]
                points = []

            # accumulate forces
            for (a, b) in segs:
                ax, ay = a; bx, by = b
                dfx, dfy = add_segment_force(px, py, ax, ay, bx, by)
                fx += dfx; fy += dfy

            if 'points' in locals():
                for (qx, qy) in points:
                    dfx, dfy = add_point_force(px, py, qx, qy)
                    fx += dfx; fy += dfy

            out[i, 0] = fx
            out[i, 1] = fy

        return out

    def _compute_force(
        self,
        direction_x: torch.Tensor,
        direction_y: torch.Tensor,
        distance: torch.Tensor,
        A: float, 
        B: float,
        mask: torch.Tensor
    ) -> torch.Tensor:
        force_magnitude = A * torch.exp(-distance / B)
        force_magnitude *= mask
        total_force_x = -torch.sum(direction_x * force_magnitude, dim=1)
        total_force_y = -torch.sum(direction_y * force_magnitude, dim=1)
        return torch.stack((total_force_x, total_force_y), dim=-1)

    @staticmethod
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "SocialForceB160":
        with open(path, "r") as file:
            param_grid = json.load(file)
        model = SocialForceB160(**param_grid)
        model.to(device)
        return model

    def save_model(self, path: str) -> None:
        param_dict = {
            'A_interaction': self.A_interaction.item(),
            'A_obstacle': self.A_obstacle.item(),
            'B_interaction': self.B_interaction.item(),
            'B_obstacle': self.B_obstacle.item(),
            'tau': self.tau.item(),
            'desired_speed': self.desired_speed.item()
        }
        with open(path + '.json', 'w') as f:
            json.dump(param_dict, f, indent=4)

    @staticmethod
    def tune(
        scenes: dict[str, Scene],
        param_grid: dict[str, list],
        device: str = "cpu",
        goal_radius: float = 0.4,
        steps: int = 300,
        fdm_win_size: int = 20,
        metric: str = "ADE"
    ) -> "SocialForceB160":
        from evaluation.evaluator import Evaluator
        from models.predictor import Predictor

        best_score = float("inf")
        best_params = None
        keys = list(param_grid.keys()); values = list(param_grid.values())

        print("Starting grid search for SocialForceB160 tuning...")
        for combo in tqdm(list(itertools.product(*values))):
            param_set = dict(zip(keys, combo))
            model = SocialForceB160(**param_set)
            predictor = Predictor(model=model, device=device)

            total_score = 0.0
            for scene in scenes.values():
                simulated = scene.simulate(
                    predict_acc_func=predictor.predict,
                    total_steps=steps,
                    goal_radius=goal_radius
                )
                simulated = simulated.approximate_velocities(fdm_win_size, "central")
                simulated = simulated.approximate_accelerations(fdm_win_size, "central")
                evaluator = Evaluator()
                eval_result = evaluator.evaluate_ade_fde(scene_gt=scene, scene_pred=simulated)
                total_score += eval_result[metric]

            avg_score = total_score / max(1, len(scenes))
            if avg_score < best_score:
                best_score = avg_score
                best_params = param_set

        print(f"Best {metric}: {best_score:.4f} with params: {best_params}")
        return SocialForceB160(**best_params)
