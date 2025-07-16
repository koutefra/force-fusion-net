class ComprehensiveEvaluator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        
    def evaluate_trajectory_metrics(self, scene_gt, scene_pred, scene_name: str, model_name: str) -> Dict[str, float]:
        """Calculate trajectory-based metrics"""
        evaluator = Evaluator()
        basic_metrics = evaluator.evaluate_scene(scene_pred)
        
        # Additional custom metrics
        metrics = {}
        
        # Extract trajectories
        gt_trajectories = self._extract_trajectories(scene_gt)
        pred_trajectories = self._extract_trajectories(scene_pred)
        
        # Average Displacement Error (ADE)
        ade_values = []
        # Final Displacement Error (FDE)
        fde_values = []
        
        for person_id in gt_trajectories:
            if person_id in pred_trajectories:
                gt_traj = np.array(gt_trajectories[person_id])
                pred_traj = np.array(pred_trajectories[person_id])
                
                # Align trajectories (take minimum length)
                min_len = min(len(gt_traj), len(pred_traj))
                gt_traj = gt_traj[:min_len]
                pred_traj = pred_traj[:min_len]
                
                if min_len > 0:
                    # ADE: Average displacement over all time steps
                    displacements = np.linalg.norm(gt_traj - pred_traj, axis=1)
                    ade_values.append(np.mean(displacements))
                    
                    # FDE: Final displacement error
                    fde_values.append(displacements[-1])
        
        metrics['ADE'] = np.mean(ade_values) if ade_values else float('inf')
        metrics['FDE'] = np.mean(fde_values) if fde_values else float('inf')
        metrics['ADE_std'] = np.std(ade_values) if ade_values else 0.0
        metrics['FDE_std'] = np.std(fde_values) if fde_values else 0.0
        
        # Add basic metrics
        if hasattr(basic_metrics, '__dict__'):
            for key, value in basic_metrics.__dict__.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
        
        return metrics
    
    def evaluate_flow_metrics(self, scene_gt, scene_pred, scene_name: str, model_name: str) -> Dict[str, float]:
        """Calculate flow-related metrics specific to bottleneck scenarios"""
        metrics = {}
        
        # Extract flow rates (people per second through bottleneck)
        gt_flow_rate = self._calculate_flow_rate(scene_gt)
        pred_flow_rate = self._calculate_flow_rate(scene_pred)
        
        metrics['flow_rate_gt'] = gt_flow_rate
        metrics['flow_rate_pred'] = pred_flow_rate
        metrics['flow_rate_error'] = abs(gt_flow_rate - pred_flow_rate)
        metrics['flow_rate_relative_error'] = abs(gt_flow_rate - pred_flow_rate) / max(gt_flow_rate, 1e-6)
        
        # Average transit time
        gt_transit_times = self._calculate_transit_times(scene_gt)
        pred_transit_times = self._calculate_transit_times(scene_pred)
        
        metrics['avg_transit_time_gt'] = np.mean(gt_transit_times) if gt_transit_times else 0
        metrics['avg_transit_time_pred'] = np.mean(pred_transit_times) if pred_transit_times else 0
        metrics['transit_time_error'] = abs(metrics['avg_transit_time_gt'] - metrics['avg_transit_time_pred'])
        
        return metrics
    
    def evaluate_collision_metrics(self, scene_pred, scene_name: str, model_name: str) -> Dict[str, float]:
        """Calculate collision-related metrics"""
        metrics = {}
        
        collision_count = self._count_collisions(scene_pred)
        total_interactions = self._count_total_interactions(scene_pred)
        
        metrics['collision_count'] = collision_count
        metrics['collision_rate'] = collision_count / max(total_interactions, 1)
        
        return metrics
    
    def evaluate_density_metrics(self, scene_gt, scene_pred, scene_name: str, model_name: str) -> Dict[str, float]:
        """Calculate density distribution metrics"""
        metrics = {}
        
        # Calculate density distributions in bottleneck area
        gt_densities = self._calculate_density_distribution(scene_gt)
        pred_densities = self._calculate_density_distribution(scene_pred)
        
        # KL divergence between density distributions
        kl_div = self._calculate_kl_divergence(gt_densities, pred_densities)
        metrics['density_kl_divergence'] = kl_div
        
        # Average density in bottleneck
        metrics['avg_density_gt'] = np.mean(gt_densities) if len(gt_densities) > 0 else 0
        metrics['avg_density_pred'] = np.mean(pred_densities) if len(pred_densities) > 0 else 0
        
        return metrics
    
    def _extract_trajectories(self, scene) -> Dict[int, List[Tuple[float, float]]]:
        """Extract trajectories from scene"""
        trajectories = {}
        for frame in scene.frames:
            for person in frame.people:
                if person.id not in trajectories:
                    trajectories[person.id] = []
                trajectories[person.id].append((person.position.x, person.position.y))
        return trajectories
    
    def _calculate_flow_rate(self, scene) -> float:
        """Calculate flow rate through bottleneck"""
        # This is a simplified calculation - adjust based on your scene geometry
        # Count people who successfully pass through the bottleneck
        completed_passages = 0
        total_time = len(scene.frames) * 0.04  # Assuming 25 FPS
        
        # You might need to adjust this based on your specific bottleneck geometry
        for person_id in self._extract_trajectories(scene):
            traj = self._extract_trajectories(scene)[person_id]
            if len(traj) > 1:
                start_pos = traj[0]
                end_pos = traj[-1]
                # Check if person moved from one side to the other (simplified)
                if end_pos[0] > start_pos[0] + 2.0:  # Adjust threshold as needed
                    completed_passages += 1
        
        return completed_passages / total_time if total_time > 0 else 0
    
    def _calculate_transit_times(self, scene) -> List[float]:
        """Calculate transit times for each person"""
        transit_times = []
        trajectories = self._extract_trajectories(scene)
        
        for person_id, traj in trajectories.items():
            if len(traj) > 1:
                # Simple transit time = number of frames * time per frame
                transit_time = len(traj) * 0.04  # 25 FPS
                transit_times.append(transit_time)
        
        return transit_times
    
    def _count_collisions(self, scene) -> int:
        """Count number of collisions (simplified)"""
        collision_count = 0
        collision_threshold = 0.3  # meters
        
        for frame in scene.frames:
            people = frame.people
            for i, person1 in enumerate(people):
                for j, person2 in enumerate(people[i+1:], i+1):
                    distance = np.sqrt((person1.position.x - person2.position.x)**2 + 
                                     (person1.position.y - person2.position.y)**2)
                    if distance < collision_threshold:
                        collision_count += 1
        
        return collision_count
    
    def _count_total_interactions(self, scene) -> int:
        """Count total possible interactions"""
        total_interactions = 0
        for frame in scene.frames:
            n_people = len(frame.people)
            total_interactions += n_people * (n_people - 1) // 2
        return total_interactions
    
    def _calculate_density_distribution(self, scene) -> List[float]:
        """Calculate density distribution in bottleneck area"""
        densities = []
        # Define bottleneck area (adjust based on your geometry)
        bottleneck_bounds = {'x_min': 2, 'x_max': 6, 'y_min': 2, 'y_max': 6}
        area = (bottleneck_bounds['x_max'] - bottleneck_bounds['x_min']) * \
               (bottleneck_bounds['y_max'] - bottleneck_bounds['y_min'])
        
        for frame in scene.frames:
            people_in_bottleneck = 0
            for person in frame.people:
                if (bottleneck_bounds['x_min'] <= person.position.x <= bottleneck_bounds['x_max'] and
                    bottleneck_bounds['y_min'] <= person.position.y <= bottleneck_bounds['y_max']):
                    people_in_bottleneck += 1
            
            density = people_in_bottleneck / area
            densities.append(density)
        
        return densities
    
    def _calculate_kl_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate KL divergence between two distributions"""
        if len(p) == 0 or len(q) == 0:
            return float('inf')
        
        # Convert to probability distributions
        p = np.array(p) + 1e-8  # Add small epsilon to avoid log(0)
        q = np.array(q) + 1e-8
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Ensure same length
        min_len = min(len(p), len(q))
        p = p[:min_len]
        q = q[:min_len]
        
        return np.sum(p * np.log(p / q))
    
    def run_evaluation(self, args):
        """Run comprehensive evaluation"""
        np.random.seed(args.seed)
        
        # Load test dataset
        test_paths_names = [(os.path.join(args.dataset_path, name + '.txt'), name) 
                           for name in args.test_scenes]
        test_dataset = SceneDataset.from_loaders([JulichCaserneLoader(test_paths_names)])
        test_dataset = test_dataset.approximate_velocities(args.fdm_win_size, "backward")
        
        # Prepare model names
        if args.model_names:
            model_names = args.model_names
        else:
            model_names = [f"{model_type}_{i}" for i, model_type in enumerate(args.model_types)]
        
        # Load models
        models = {}
        for model_path, model_type, model_name in zip(args.model_paths, args.model_types, model_names):
            if model_type == 'direct_net':
                model = DirectNet.from_weight_file(model_path)
            elif model_type == 'fusion_net':
                model = FusionNet.from_weight_file(model_path)
            elif model_type == 'social_force':
                model = SocialForce.from_weight_file(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            predictor = Predictor(model, device=args.device)
            models[model_name] = predictor
        
        # Add baseline models
        if 'social_force' in args.baseline_comparisons:
            models['SocialForce_baseline'] = Predictor(SocialForce(), device=args.device)
        
        all_results = {}
        
        # Evaluate each scene with each model
        for scene_name, scene_gt in test_dataset.scenes.items():
            print(f"Evaluating scene: {scene_name}")
            scene_results = {}
            
            # Ground truth reference
            scene_gt_sim = scene_gt.take_first_n_frames(args.simulation_steps)
            
            for model_name, predictor in models.items():
                print(f"  Model: {model_name}")
                
                # Run simulation
                if model_name == 'GroundTruth' or 'gt' in args.baseline_comparisons:
                    scene_pred = scene_gt_sim
                else:
                    scene_pred = scene_gt.simulate(
                        predict_acc_func=predictor.predict,
                        total_steps=args.simulation_steps,
                        goal_radius=args.goal_radius
                    )
                
                # Calculate all metrics
                trajectory_metrics = self.evaluate_trajectory_metrics(scene_gt_sim, scene_pred, scene_name, model_name)
                flow_metrics = self.evaluate_flow_metrics(scene_gt_sim, scene_pred, scene_name, model_name)
                collision_metrics = self.evaluate_collision_metrics(scene_pred, scene_name, model_name)
                density_metrics = self.evaluate_density_metrics(scene_gt_sim, scene_pred, scene_name, model_name)
                
                # Combine all metrics
                all_metrics = {**trajectory_metrics, **flow_metrics, **collision_metrics, **density_metrics}
                scene_results[model_name] = all_metrics
            
            all_results[scene_name] = scene_results
        
        # Save results
        self.save_results(all_results, args)
        self.create_visualizations(all_results, args)
        self.create_summary_report(all_results, args)
        
        return all_results
    
    def save_results(self, results: Dict, args):
        """Save results to files"""
        # Save as JSON
        with open(os.path.join(self.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV for easy analysis
        rows = []
        for scene_name, scene_results in results.items():
            for model_name, metrics in scene_results.items():
                row = {'scene': scene_name, 'model': model_name}
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "evaluation_results.csv"), index=False)
        print(f"Results saved to {self.output_dir}")
    
    def create_visualizations(self, results: Dict, args):
        """Create visualization plots"""
        # Convert to DataFrame for easier plotting
        rows = []
        for scene_name, scene_results in results.items():
            for model_name, metrics in scene_results.items():
                row = {'scene': scene_name, 'model': model_name}
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Key metrics to visualize
        key_metrics = ['ADE', 'FDE', 'flow_rate_error', 'collision_rate', 'transit_time_error']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if metric in df.columns:
                sns.boxplot(data=df, x='model', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric} by Model')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scene-specific performance
        if 'ADE' in df.columns:
            plt.figure(figsize=(12, 8))
            pivot_df = df.pivot(index='scene', columns='model', values='ADE')
            sns.heatmap(pivot_df, annot=True, cmap='viridis_r', fmt='.3f')
            plt.title('ADE by Scene and Model')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "ade_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_report(self, results: Dict, args):
        """Create a summary report"""
        # Convert to DataFrame
        rows = []
        for scene_name, scene_results in results.items():
            for model_name, metrics in scene_results.items():
                row = {'scene': scene_name, 'model': model_name}
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Calculate summary statistics
        summary_stats = df.groupby('model').agg({
            'ADE': ['mean', 'std'],
            'FDE': ['mean', 'std'],
            'flow_rate_error': ['mean', 'std'],
            'collision_rate': ['mean', 'std'],
            'transit_time_error': ['mean', 'std']
        }).round(4)
        
        # Save summary report
        with open(os.path.join(self.output_dir, "summary_report.txt"), "w") as f:
            f.write("PEDESTRIAN MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test scenes: {', '.join(args.test_scenes)}\n")
            f.write(f"Models evaluated: {', '.join(df['model'].unique())}\n")
            f.write(f"Simulation steps: {args.simulation_steps}\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(summary_stats.to_string())
            f.write("\n\n")
            
            # Best performing model for each metric
            f.write("BEST PERFORMING MODELS\n")
            f.write("-" * 30 + "\n")
            for metric in ['ADE', 'FDE', 'flow_rate_error', 'collision_rate']:
                if metric in df.columns:
                    best_model = df.groupby('model')[metric].mean().idxmin()
                    best_value = df.groupby('model')[metric].mean().min()
                    f.write(f"{metric}: {best_model} ({best_value:.4f})\n")
        
        print(f"Summary report saved to {os.path.join(self.output_dir, 'summary_report.txt')}")