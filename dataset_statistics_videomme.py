import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

class VideoFrameAnalyzer:
    def __init__(self, dataset_path, frames_path, scores_path):
        """
        Initialize the analyzer with paths to dataset and frame/score files
        
        Args:
            dataset_path: Path to the main JSON dataset file
            frames_path: Path to single frames.json file containing all video frames
            scores_path: Path to single scores.json file containing all video scores
        """
        self.dataset_path = dataset_path
        self.frames_path = Path(frames_path)
        self.scores_path = Path(scores_path)
        self.dataset = []
        self.all_frames = None
        self.all_scores = None
        self.results = []
        self.global_frame_stats = None
        
    def load_dataset(self):
        """Load the main dataset"""
        with open(self.dataset_path, 'r') as f:
            content = f.read()
            if content.strip().startswith('['):
                self.dataset = json.loads(content)
            else:
                # JSON lines format
                self.dataset = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        print(f"Loaded {len(self.dataset)} entries from dataset")
        
    def load_all_frames_and_scores(self):
        """Load all frames and scores from single JSON files"""
        print("Loading frames and scores...")
        
        try:
            with open(self.frames_path, 'r') as f:
                self.all_frames = json.load(f)
            print(f"Loaded frames for {len(self.all_frames)} videos")
        except FileNotFoundError:
            print(f"Error: Frames file not found at {self.frames_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in frames file: {e}")
            return False
            
        try:
            with open(self.scores_path, 'r') as f:
                self.all_scores = json.load(f)
            print(f"Loaded scores for {len(self.all_scores)} videos")
        except FileNotFoundError:
            print(f"Error: Scores file not found at {self.scores_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in scores file: {e}")
            return False
        
        return True
    
    def calculate_global_frame_statistics(self):
        """Calculate statistics about all frames across all videos"""
        print("\nCalculating global frame statistics...")
        
        all_scores_flat = []
        total_frames = 0
        video_count = 0
        
        # Collect all scores from all videos
        if isinstance(self.all_scores, dict):
            for video_id, scores in self.all_scores.items():
                if isinstance(scores, list):
                    if len(scores) > 0 and isinstance(scores[0], list):
                        scores = scores[0]
                    all_scores_flat.extend(scores)
                    total_frames += len(scores)
                    video_count += 1
        elif isinstance(self.all_scores, list):
            for scores in self.all_scores:
                if isinstance(scores, list):
                    if len(scores) > 0 and isinstance(scores[0], list):
                        scores = scores[0]
                    all_scores_flat.extend(scores)
                    total_frames += len(scores)
                    video_count += 1
        
        all_scores_array = np.array(all_scores_flat)
        
        # Calculate percentiles
        percentiles = [50, 70, 80, 90, 95, 99]
        percentile_thresholds = {p: np.percentile(all_scores_array, p) for p in percentiles}
        
        # Count frames in each percentile range
        percentile_counts = {}
        percentile_counts['total'] = len(all_scores_array)
        
        for p in percentiles:
            count = np.sum(all_scores_array >= percentile_thresholds[p])
            percentile_counts[f'top_{p}%'] = count
            percentile_counts[f'top_{p}%_percentage'] = (count / len(all_scores_array) * 100)
        
        # Calculate score distribution statistics
        score_stats = {
            'total_frames': int(total_frames),
            'total_videos': int(video_count),
            'avg_frames_per_video': float(total_frames / video_count) if video_count > 0 else 0,
            'score_min': float(np.min(all_scores_array)),
            'score_max': float(np.max(all_scores_array)),
            'score_mean': float(np.mean(all_scores_array)),
            'score_median': float(np.median(all_scores_array)),
            'score_std': float(np.std(all_scores_array)),
            'percentile_thresholds': {k: float(v) for k, v in percentile_thresholds.items()},
            'percentile_counts': percentile_counts,
        }
        
        # Calculate distribution by score ranges
        score_ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        range_counts = []
        for i in range(len(score_ranges) - 1):
            count = np.sum((all_scores_array >= score_ranges[i]) & (all_scores_array < score_ranges[i+1]))
            range_counts.append({
                'range': f'{score_ranges[i]:.1f}-{score_ranges[i+1]:.1f}',
                'count': int(count),
                'percentage': float(count / len(all_scores_array) * 100)
            })
        # Add the max value
        count_max = np.sum(all_scores_array == 1.0)
        if count_max > 0:
            range_counts[-1]['count'] += int(count_max)
            range_counts[-1]['percentage'] = float(range_counts[-1]['count'] / len(all_scores_array) * 100)
        
        score_stats['score_distribution'] = range_counts
        
        self.global_frame_stats = score_stats
        
        return score_stats
    
    def get_frames_and_scores(self, video_id):
        """Get frames and scores for a specific video from loaded data"""
        # Handle both string and integer video IDs
        video_id_str = str(video_id)
        
        # Try different formats
        frames = None
        scores = None
        
        # Check if data is a list (indexed by position) or dict (indexed by video_id)
        if isinstance(self.all_frames, list):
            # Assuming video_id corresponds to index
            try:
                idx = int(video_id) - 1  # If IDs start at 1
                if 0 <= idx < len(self.all_frames):
                    frames = self.all_frames[idx]
                    scores = self.all_scores[idx]
            except (ValueError, IndexError):
                pass
        elif isinstance(self.all_frames, dict):
            # Data is indexed by video_id as key
            frames = self.all_frames.get(video_id_str)
            scores = self.all_scores.get(video_id_str)
            
            # Try without leading zeros
            if frames is None:
                frames = self.all_frames.get(video_id_str.lstrip('0'))
                scores = self.all_scores.get(video_id_str.lstrip('0'))
            
            # Try with leading zeros (3 digits)
            if frames is None:
                padded_id = video_id_str.zfill(3)
                frames = self.all_frames.get(padded_id)
                scores = self.all_scores.get(padded_id)
        
        return frames, scores
    
    def calculate_percentile_stats(self, selected_frames, all_frames, all_scores):
        """
        Calculate which percentile each selected frame falls into
        
        Args:
            selected_frames: List of frame indices from frame_idx
            all_frames: All frame numbers for this video
            all_scores: All scores for this video
        """
        # Flatten if nested
        if isinstance(all_frames, list) and len(all_frames) > 0 and isinstance(all_frames[0], list):
            all_frames = all_frames[0]
        if isinstance(all_scores, list) and len(all_scores) > 0 and isinstance(all_scores[0], list):
            all_scores = all_scores[0]
        
        # Ensure we have valid data
        if not all_frames or not all_scores:
            return None
        
        if len(all_frames) != len(all_scores):
            print(f"Warning: Frame and score lengths don't match ({len(all_frames)} vs {len(all_scores)})")
            return None
        
        # Create frame to score mapping
        frame_score_map = dict(zip(all_frames, all_scores))
        
        # Get scores for selected frames
        selected_scores = []
        for f in selected_frames:
            score = frame_score_map.get(int(f))
            if score is None:
                # Try to find closest frame
                closest_frame = min(all_frames, key=lambda x: abs(x - int(f)))
                score = frame_score_map.get(closest_frame)
            selected_scores.append(score)
        
        # Calculate global statistics for this video
        all_scores_array = np.array(all_scores)
        max_score = np.max(all_scores_array)
        min_score = np.min(all_scores_array)
        mean_score = np.mean(all_scores_array)
        median_score = np.median(all_scores_array)
        
        # Calculate percentiles
        percentiles = [50, 70, 80, 90, 95, 99]
        percentile_thresholds = {p: np.percentile(all_scores_array, p) for p in percentiles}
        
        # Analyze selected frames
        valid_scores = [s for s in selected_scores if s is not None]
        selected_stats = {
            'selected_scores': selected_scores,
            'selected_max': max(valid_scores) if valid_scores else 0,
            'selected_min': min(valid_scores) if valid_scores else 0,
            'selected_mean': np.mean(valid_scores) if valid_scores else 0,
        }
        
        # Count frames in each percentile
        percentile_counts = {p: 0 for p in percentiles}
        for score in selected_scores:
            if score is not None:
                for p in percentiles:
                    if score >= percentile_thresholds[p]:
                        percentile_counts[p] += 1
        
        # Calculate percentages
        total_selected = len(valid_scores)
        percentile_percentages = {p: (count / total_selected * 100) if total_selected > 0 else 0 
                                   for p, count in percentile_counts.items()}
        
        return {
            'global_stats': {
                'max': float(max_score),
                'min': float(min_score),
                'mean': float(mean_score),
                'median': float(median_score),
                'total_frames': len(all_scores),
            },
            'percentile_thresholds': {k: float(v) for k, v in percentile_thresholds.items()},
            'selected_stats': selected_stats,
            'percentile_counts': percentile_counts,
            'percentile_percentages': percentile_percentages,
            'total_selected_frames': total_selected,
        }
    
    def analyze_dataset(self):
        """Analyze the entire dataset"""
        skipped = 0
        
        for entry in self.dataset:
            video_id = entry['video_id']
            frames, scores = self.get_frames_and_scores(video_id)
            
            if frames is None or scores is None:
                skipped += 1
                continue
            
            frame_idx = entry.get('frame_idx', [])
            if not frame_idx:
                skipped += 1
                continue
            
            stats = self.calculate_percentile_stats(frame_idx, frames, scores)
            
            if stats is None:
                skipped += 1
                continue
            
            result = {
                'video_id': video_id,
                'duration': entry.get('duration', 'unknown'),
                'domain': entry.get('domain', 'unknown'),
                'sub_category': entry.get('sub_category', 'unknown'),
                'task_type': entry.get('task_type', 'unknown'),
                'num_frames': len(frame_idx),
                **stats
            }
            
            self.results.append(result)
        
        print(f"Analyzed {len(self.results)} videos successfully")
        if skipped > 0:
            print(f"Skipped {skipped} videos due to missing or invalid data")
    
    def generate_summary_statistics(self):
        """Generate summary statistics across different categories"""
        if not self.results:
            print("Warning: No results to generate statistics from")
            return {'overall': {}}
        
        df = pd.DataFrame(self.results)
        
        summaries = {}
        
        # Overall statistics
        summaries['overall'] = self._calculate_group_stats(df)
        
        # By duration
        if 'duration' in df.columns:
            summaries['by_duration'] = {}
            for duration in df['duration'].unique():
                duration_df = df[df['duration'] == duration]
                summaries['by_duration'][duration] = self._calculate_group_stats(duration_df)
        
        # By domain
        if 'domain' in df.columns:
            summaries['by_domain'] = {}
            for domain in df['domain'].unique():
                domain_df = df[df['domain'] == domain]
                summaries['by_domain'][domain] = self._calculate_group_stats(domain_df)
        
        # By task type
        if 'task_type' in df.columns:
            summaries['by_task_type'] = {}
            for task in df['task_type'].unique():
                task_df = df[df['task_type'] == task]
                summaries['by_task_type'][task] = self._calculate_group_stats(task_df)
        
        return summaries
    
    def _calculate_group_stats(self, df):
        """Calculate statistics for a group of videos"""
        if len(df) == 0:
            return {}
        
        stats = {
            'count': len(df),
            'avg_num_frames': df['num_frames'].mean(),
            'percentile_coverage': {}
        }
        
        # Average percentile coverage
        for p in [50, 70, 80, 90, 95, 99]:
            percentages = [r['percentile_percentages'].get(p, 0) for r in df.to_dict('records')]
            stats['percentile_coverage'][f'top_{p}%'] = {
                'mean': np.mean(percentages),
                'std': np.std(percentages),
                'min': np.min(percentages),
                'max': np.max(percentages),
            }
        
        return stats
    
    def create_visualizations(self, output_dir='output'):
        """Create various visualizations"""
        if not self.results:
            print("Warning: No results to visualize")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        df = pd.DataFrame(self.results)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Percentile coverage heatmap by duration
        self._plot_percentile_heatmap_by_category(df, 'duration', output_path)
        
        # 2. Distribution of frames in top percentiles
        self._plot_percentile_distribution(df, output_path)
        
        # 3. Box plots by duration
        self._plot_boxplots_by_duration(df, output_path)
        
        # 4. Scatter plot: number of frames vs coverage
        self._plot_frames_vs_coverage(df, output_path)
        
        # 5. Bar chart comparison across categories
        self._plot_category_comparison(df, output_path)
        
        # 6. Global frame distribution visualizations
        if self.global_frame_stats:
            self._plot_global_score_distribution(output_path)
            self._plot_global_percentile_distribution(output_path)
        
        print(f"Visualizations saved to {output_path}")
    
    def _plot_global_score_distribution(self, output_path):
        """Plot global score distribution across all frames"""
        if not self.global_frame_stats:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Score range distribution (bar chart)
        dist_data = self.global_frame_stats['score_distribution']
        ranges = [d['range'] for d in dist_data]
        counts = [d['count'] for d in dist_data]
        percentages = [d['percentage'] for d in dist_data]
        
        ax1.bar(ranges, counts, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Score Range')
        ax1.set_ylabel('Number of Frames')
        ax1.set_title(f'Global Frame Distribution by Score Range\n(Total: {self.global_frame_stats["total_frames"]:,} frames)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax1.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
        
        # Percentile distribution (horizontal bar chart)
        percentiles = [50, 70, 80, 90, 95, 99]
        counts_pct = [self.global_frame_stats['percentile_counts'][f'top_{p}%'] for p in percentiles]
        percentages_pct = [self.global_frame_stats['percentile_counts'][f'top_{p}%_percentage'] for p in percentiles]
        
        y_pos = np.arange(len(percentiles))
        ax2.barh(y_pos, counts_pct, alpha=0.7, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Top {p}%' for p in percentiles])
        ax2.set_xlabel('Number of Frames')
        ax2.set_title('Global Frame Count by Percentile Threshold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add labels
        for i, (count, pct) in enumerate(zip(counts_pct, percentages_pct)):
            ax2.text(count, i, f'  {count:,} ({pct:.1f}%)', va='center')
        
        plt.tight_layout()
        plt.savefig(output_path / 'global_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_global_percentile_distribution(self, output_path):
        """Plot comparison between global frame availability and selection coverage"""
        if not self.global_frame_stats:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        percentiles = [50, 70, 80, 90, 95, 99]
        
        # Global availability (how many frames exist in each percentile)
        global_percentages = [self.global_frame_stats['percentile_counts'][f'top_{p}%_percentage'] for p in percentiles]
        
        # Average selection coverage (how many selected frames are in each percentile)
        df = pd.DataFrame(self.results)
        selection_percentages = []
        for p in percentiles:
            percentages = [r['percentile_percentages'].get(p, 0) for r in df.to_dict('records')]
            selection_percentages.append(np.mean(percentages))
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, global_percentages, width, label='Global Availability (%)', alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, selection_percentages, width, label='Avg Selection Coverage (%)', alpha=0.8, color='coral', edgecolor='black')
        
        ax.set_xlabel('Percentile Threshold', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Global Frame Availability vs Selection Coverage by Percentile', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Top {p}%' for p in percentiles])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'global_vs_selection_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_percentile_heatmap_by_category(self, df, category, output_path):
        """Create heatmap showing percentile coverage by category"""
        percentiles = [50, 70, 80, 90, 95, 99]
        categories = df[category].unique()
        
        data = []
        for cat in categories:
            cat_df = df[df[category] == cat]
            row = []
            for p in percentiles:
                percentages = [r['percentile_percentages'].get(p, 0) for r in cat_df.to_dict('records')]
                row.append(np.mean(percentages))
            data.append(row)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt='.1f', xticklabels=[f'Top {p}%' for p in percentiles],
                    yticklabels=categories, cmap='YlOrRd', cbar_kws={'label': 'Percentage of Frames'})
        plt.title(f'Average Frame Coverage in Top Percentiles by {category.capitalize()}')
        plt.xlabel('Percentile Threshold')
        plt.ylabel(category.capitalize())
        plt.tight_layout()
        plt.savefig(output_path / f'heatmap_{category}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_percentile_distribution(self, df, output_path):
        """Plot distribution of frames across percentiles"""
        percentiles = [50, 70, 80, 90, 95, 99]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, p in enumerate(percentiles):
            percentages = [r['percentile_percentages'].get(p, 0) for r in df.to_dict('records')]
            axes[idx].hist(percentages, bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Top {p}% Coverage')
            axes[idx].set_xlabel('Percentage of Selected Frames')
            axes[idx].set_ylabel('Number of Videos')
            axes[idx].axvline(np.mean(percentages), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(percentages):.1f}%')
            axes[idx].legend()
        
        plt.suptitle('Distribution of Frame Coverage Across Percentiles')
        plt.tight_layout()
        plt.savefig(output_path / 'percentile_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplots_by_duration(self, df, output_path):
        """Create box plots for percentile coverage by duration"""
        percentiles = [90, 95, 99]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, p in enumerate(percentiles):
            data_by_duration = []
            labels = []
            
            for duration in df['duration'].unique():
                duration_df = df[df['duration'] == duration]
                percentages = [r['percentile_percentages'].get(p, 0) for r in duration_df.to_dict('records')]
                data_by_duration.append(percentages)
                labels.append(duration)
            
            axes[idx].boxplot(data_by_duration, labels=labels)
            axes[idx].set_title(f'Top {p}% Coverage by Duration')
            axes[idx].set_xlabel('Duration')
            axes[idx].set_ylabel('Percentage of Frames')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'boxplots_by_duration.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_frames_vs_coverage(self, df, output_path):
        """Scatter plot: number of frames vs top percentile coverage"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top 95%
        percentages_95 = [r['percentile_percentages'].get(95, 0) for r in df.to_dict('records')]
        axes[0].scatter(df['num_frames'], percentages_95, alpha=0.6, c=df['duration'].astype('category').cat.codes)
        axes[0].set_xlabel('Number of Selected Frames')
        axes[0].set_ylabel('% of Frames in Top 95%')
        axes[0].set_title('Frame Count vs Top 95% Coverage')
        axes[0].grid(True, alpha=0.3)
        
        # Top 99%
        percentages_99 = [r['percentile_percentages'].get(99, 0) for r in df.to_dict('records')]
        axes[1].scatter(df['num_frames'], percentages_99, alpha=0.6, c=df['duration'].astype('category').cat.codes)
        axes[1].set_xlabel('Number of Selected Frames')
        axes[1].set_ylabel('% of Frames in Top 99%')
        axes[1].set_title('Frame Count vs Top 99% Coverage')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'frames_vs_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_comparison(self, df, output_path):
        """Bar chart comparing average coverage across categories"""
        percentiles = [90, 95, 99]
        categories = ['duration', 'domain', 'task_type']
        
        for category in categories:
            if category not in df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            cat_values = df[category].unique()
            x = np.arange(len(cat_values))
            width = 0.25
            
            for idx, p in enumerate(percentiles):
                means = []
                for cat in cat_values:
                    cat_df = df[df[category] == cat]
                    percentages = [r['percentile_percentages'].get(p, 0) for r in cat_df.to_dict('records')]
                    means.append(np.mean(percentages))
                
                ax.bar(x + idx * width, means, width, label=f'Top {p}%')
            
            ax.set_xlabel(category.capitalize())
            ax.set_ylabel('Average % of Frames')
            ax.set_title(f'Frame Coverage Comparison by {category.capitalize()}')
            ax.set_xticks(x + width)
            ax.set_xticklabels(cat_values, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_path / f'comparison_{category}.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_tables(self, output_dir='output'):
        """Generate summary tables"""
        if not self.results:
            print("Warning: No results to generate tables from")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        summaries = self.generate_summary_statistics()
        
        # Overall summary table
        self._create_summary_table(summaries['overall'], 'Overall', output_path)
        
        # Duration summary
        if 'by_duration' in summaries:
            for duration, stats in summaries['by_duration'].items():
                self._create_summary_table(stats, f'Duration: {duration}', output_path)
        
        # Create comparison table
        self._create_comparison_table(summaries, output_path)
        
        # Create global statistics table
        if self.global_frame_stats:
            self._create_global_stats_table(output_path)
        
        print(f"Tables saved to {output_path}")
    
    def _create_global_stats_table(self, output_path):
        """Create table for global frame statistics"""
        if not self.global_frame_stats:
            return
        
        # Overall statistics table
        overall_data = {
            'Metric': ['Total Frames', 'Total Videos', 'Avg Frames per Video', 
                      'Min Score', 'Max Score', 'Mean Score', 'Median Score', 'Std Dev Score'],
            'Value': [
                f"{self.global_frame_stats['total_frames']:,}",
                f"{self.global_frame_stats['total_videos']:,}",
                f"{self.global_frame_stats['avg_frames_per_video']:.2f}",
                f"{self.global_frame_stats['score_min']:.4f}",
                f"{self.global_frame_stats['score_max']:.4f}",
                f"{self.global_frame_stats['score_mean']:.4f}",
                f"{self.global_frame_stats['score_median']:.4f}",
                f"{self.global_frame_stats['score_std']:.4f}",
            ]
        }
        df_overall = pd.DataFrame(overall_data)
        df_overall.to_csv(output_path / 'table_global_overall_stats.csv', index=False)
        
        # Percentile thresholds table
        percentile_data = []
        for p, threshold in self.global_frame_stats['percentile_thresholds'].items():
            count = self.global_frame_stats['percentile_counts'][f'top_{p}%']
            percentage = self.global_frame_stats['percentile_counts'][f'top_{p}%_percentage']
            percentile_data.append({
                'Percentile': f'Top {p}%',
                'Score Threshold': f'{threshold:.4f}',
                'Frame Count': f'{count:,}',
                'Percentage': f'{percentage:.2f}%'
            })
        
        df_percentile = pd.DataFrame(percentile_data)
        df_percentile.to_csv(output_path / 'table_global_percentile_stats.csv', index=False)
        
        # Score distribution table
        dist_data = []
        for item in self.global_frame_stats['score_distribution']:
            dist_data.append({
                'Score Range': item['range'],
                'Frame Count': f"{item['count']:,}",
                'Percentage': f"{item['percentage']:.2f}%"
            })
        
        df_dist = pd.DataFrame(dist_data)
        df_dist.to_csv(output_path / 'table_global_score_distribution.csv', index=False)
        
        # Create comprehensive HTML report
        html = "<h1>Global Frame Statistics</h1>\n"
        html += "<h2>Overall Statistics</h2>\n"
        html += df_overall.to_html(index=False)
        html += "<h2>Percentile Statistics</h2>\n"
        html += df_percentile.to_html(index=False)
        html += "<h2>Score Distribution</h2>\n"
        html += df_dist.to_html(index=False)
        
        with open(output_path / 'table_global_stats.html', 'w') as f:
            f.write(html)
    
    def _create_summary_table(self, stats, title, output_path):
        """Create a summary table for a group"""
        if not stats or 'percentile_coverage' not in stats:
            return
        
        data = []
        for percentile, values in stats['percentile_coverage'].items():
            data.append({
                'Percentile': percentile,
                'Mean Coverage (%)': f"{values['mean']:.2f}",
                'Std Dev': f"{values['std']:.2f}",
                'Min (%)': f"{values['min']:.2f}",
                'Max (%)': f"{values['max']:.2f}",
            })
        
        df_table = pd.DataFrame(data)
        
        # Save as CSV
        filename = title.replace(' ', '_').replace(':', '').lower()
        df_table.to_csv(output_path / f'table_{filename}.csv', index=False)
        
        # Save as HTML for better viewing
        html = f"<h2>{title}</h2>\n"
        html += f"<p>Total Videos: {stats['count']}, Average Frames per Video: {stats['avg_num_frames']:.2f}</p>\n"
        html += df_table.to_html(index=False)
        
        with open(output_path / f'table_{filename}.html', 'w') as f:
            f.write(html)
    
    def _create_comparison_table(self, summaries, output_path):
        """Create a comparison table across all categories"""
        data = []
        
        # Overall
        if 'overall' in summaries and summaries['overall']:
            for p in [90, 95, 99]:
                percentile_key = f'top_{p}%'
                if 'percentile_coverage' in summaries['overall'] and percentile_key in summaries['overall']['percentile_coverage']:
                    data.append({
                        'Category': 'Overall',
                        'Subcategory': 'All',
                        'Percentile': f'Top {p}%',
                        'Mean Coverage (%)': f"{summaries['overall']['percentile_coverage'][percentile_key]['mean']:.2f}",
                        'Count': summaries['overall']['count']
                    })
        
        # By duration
        if 'by_duration' in summaries:
            for duration, stats in summaries['by_duration'].items():
                if not stats:
                    continue
                for p in [90, 95, 99]:
                    percentile_key = f'top_{p}%'
                    if 'percentile_coverage' in stats and percentile_key in stats['percentile_coverage']:
                        data.append({
                            'Category': 'Duration',
                            'Subcategory': duration,
                            'Percentile': f'Top {p}%',
                            'Mean Coverage (%)': f"{stats['percentile_coverage'][percentile_key]['mean']:.2f}",
                            'Count': stats['count']
                        })
        
        # By domain
        if 'by_domain' in summaries:
            for domain, stats in summaries['by_domain'].items():
                if not stats:
                    continue
                for p in [90, 95, 99]:
                    percentile_key = f'top_{p}%'
                    if 'percentile_coverage' in stats and percentile_key in stats['percentile_coverage']:
                        data.append({
                            'Category': 'Domain',
                            'Subcategory': domain,
                            'Percentile': f'Top {p}%',
                            'Mean Coverage (%)': f"{stats['percentile_coverage'][percentile_key]['mean']:.2f}",
                            'Count': stats['count']
                        })
        
        # By task type
        if 'by_task_type' in summaries:
            for task, stats in summaries['by_task_type'].items():
                if not stats:
                    continue
                for p in [90, 95, 99]:
                    percentile_key = f'top_{p}%'
                    if 'percentile_coverage' in stats and percentile_key in stats['percentile_coverage']:
                        data.append({
                            'Category': 'Task Type',
                            'Subcategory': task,
                            'Percentile': f'Top {p}%',
                            'Mean Coverage (%)': f"{stats['percentile_coverage'][percentile_key]['mean']:.2f}",
                            'Count': stats['count']
                        })
        
        if not data:
            print("Warning: No data to create comparison table")
            return
        
        df_comparison = pd.DataFrame(data)
        df_comparison.to_csv(output_path / 'comparison_table.csv', index=False)
        
        # Create pivot table for better viewing
        if len(df_comparison) > 0:
            try:
                pivot = df_comparison.pivot_table(
                    values='Mean Coverage (%)', 
                    index=['Category', 'Subcategory'], 
                    columns='Percentile',
                    aggfunc='first'
                )
                pivot.to_csv(output_path / 'comparison_pivot.csv')
                pivot.to_html(output_path / 'comparison_pivot.html')
            except Exception as e:
                print(f"Warning: Could not create pivot table: {e}")
    
    def save_detailed_results(self, output_dir='output'):
        """Save detailed results to JSON"""
        if not self.results:
            print("Warning: No results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        # Save analysis results
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        
        # Save global statistics
        if self.global_frame_stats:
            with open(output_path / 'global_frame_stats.json', 'w') as f:
                json.dump(self.global_frame_stats, f, indent=2, cls=NumpyEncoder)
        
        print(f"Detailed results saved to {output_path / 'detailed_results.json'}")
        if self.global_frame_stats:
            print(f"Global frame statistics saved to {output_path / 'global_frame_stats.json'}")
    
def main():
    # Configure paths
    dataset_path = 'converted_frames_testig/videomme/selected_dbfp_videomme_blip_k16_alpha0.75_sup3_score_diff_minscore0.7.json'
    frames_path = 'outscores/videomme/blip/frames.json'
    scores_path = 'outscores/videomme/blip/scores.json'
    output_dir = 'statistics/statistics_videomme'
    
    # Initialize analyzer
    analyzer = VideoFrameAnalyzer(dataset_path, frames_path, scores_path)
    
    # Run analysis
    print("Loading dataset...")
    analyzer.load_dataset()
    
    print("\nLoading frames and scores...")
    if not analyzer.load_all_frames_and_scores():
        print("Error: Could not load frames and scores. Exiting.")
        return
    
    # Calculate global statistics
    global_stats = analyzer.calculate_global_frame_statistics()
    
    print("\n" + "="*60)
    print("GLOBAL FRAME STATISTICS (All Videos)")
    print("="*60)
    print(f"Total frames across all videos: {global_stats['total_frames']:,}")
    print(f"Total videos: {global_stats['total_videos']:,}")
    print(f"Average frames per video: {global_stats['avg_frames_per_video']:.2f}")
    print(f"\nScore Statistics:")
    print(f"  Min score: {global_stats['score_min']:.4f}")
    print(f"  Max score: {global_stats['score_max']:.4f}")
    print(f"  Mean score: {global_stats['score_mean']:.4f}")
    print(f"  Median score: {global_stats['score_median']:.4f}")
    print(f"  Std dev: {global_stats['score_std']:.4f}")
    
    print(f"\nGlobal Percentile Distribution:")
    for p in [50, 70, 80, 90, 95, 99]:
        threshold = global_stats['percentile_thresholds'][p]
        count = global_stats['percentile_counts'][f'top_{p}%']
        percentage = global_stats['percentile_counts'][f'top_{p}%_percentage']
        print(f"  Top {p}%: {count:,} frames ({percentage:.2f}%) with score >= {threshold:.4f}")
    
    print(f"\nScore Range Distribution:")
    for item in global_stats['score_distribution']:
        print(f"  {item['range']}: {item['count']:,} frames ({item['percentage']:.2f}%)")
    
    print("\nAnalyzing videos...")
    analyzer.analyze_dataset()
    
    if not analyzer.results:
        print("\nError: No videos were successfully analyzed. Please check:")
        print("  1. Video IDs in dataset match keys in frames.json and scores.json")
        print("  2. Frames and scores files have correct format")
        print("  3. frame_idx values in dataset are valid")
        return
    
    print("\nGenerating summary statistics...")
    summaries = analyzer.generate_summary_statistics()
    
    # Print some key statistics
    print("\n" + "="*60)
    print("SELECTION STATISTICS (Selected Frames Only)")
    print("="*60)
    if 'overall' in summaries and summaries['overall']:
        print(f"Total videos analyzed: {summaries['overall']['count']}")
        print(f"Average frames per video: {summaries['overall']['avg_num_frames']:.2f}")
        print("\nPercentile Coverage (% of selected frames in each percentile):")
        for percentile, stats in summaries['overall']['percentile_coverage'].items():
            print(f"  {percentile}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
    else:
        print("No overall statistics available")
    
    print("\n" + "="*60)
    print("BY DURATION")
    print("="*60)
    if 'by_duration' in summaries:
        for duration, stats in summaries['by_duration'].items():
            if not stats:
                continue
            print(f"\n{duration.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg frames: {stats['avg_num_frames']:.2f}")
            if 'percentile_coverage' in stats:
                print(f"  Top 90% coverage: {stats['percentile_coverage']['top_90%']['mean']:.2f}%")
                print(f"  Top 95% coverage: {stats['percentile_coverage']['top_95%']['mean']:.2f}%")
                print(f"  Top 99% coverage: {stats['percentile_coverage']['top_99%']['mean']:.2f}%")
    
    print("\n" + "="*60)
    print("BY DOMAIN")
    print("="*60)
    if 'by_domain' in summaries:
        for domain, stats in summaries['by_domain'].items():
            if not stats:
                continue
            print(f"\n{domain}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg frames: {stats['avg_num_frames']:.2f}")
            if 'percentile_coverage' in stats:
                print(f"  Top 95% coverage: {stats['percentile_coverage']['top_95%']['mean']:.2f}%")
    
    print("\n" + "="*60)
    print("BY TASK TYPE")
    print("="*60)
    if 'by_task_type' in summaries:
        for task, stats in summaries['by_task_type'].items():
            if not stats:
                continue
            print(f"\n{task}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg frames: {stats['avg_num_frames']:.2f}")
            if 'percentile_coverage' in stats:
                print(f"  Top 95% coverage: {stats['percentile_coverage']['top_95%']['mean']:.2f}%")
    
    print("\n" + "="*60)
    print("GLOBAL vs SELECTION COMPARISON")
    print("="*60)
    print("Comparing available frames vs selected frame coverage:\n")
    print(f"{'Percentile':<15} {'Available (%)':<20} {'Selected Coverage (%)':<25} {'Efficiency':<15}")
    print("-" * 75)
    for p in [50, 70, 80, 90, 95, 99]:
        available = global_stats['percentile_counts'][f'top_{p}%_percentage']
        if 'overall' in summaries and summaries['overall'] and 'percentile_coverage' in summaries['overall']:
            selected = summaries['overall']['percentile_coverage'][f'top_{p}%']['mean']
            # Efficiency: how well we're selecting from available frames
            # If available is 5% and we select 5%, efficiency is 100%
            # If available is 5% and we select 2.5%, efficiency is 50%
            efficiency = (selected / available * 100) if available > 0 else 0
            print(f"Top {p}%{'':<8} {available:<20.2f} {selected:<25.2f} {efficiency:.1f}%")
        else:
            print(f"Top {p}%{'':<8} {available:<20.2f} {'N/A':<25} {'N/A':<15}")
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    analyzer.create_visualizations(output_dir)
    
    print("\nGenerating tables...")
    analyzer.generate_tables(output_dir)
    
    print("\nSaving detailed results...")
    analyzer.save_detailed_results(output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("\n📊 Visualizations:")
    print("  - global_score_distribution.png (global frame distribution)")
    print("  - global_vs_selection_comparison.png (availability vs selection)")
    print("  - heatmap_*.png (coverage by category)")
    print("  - percentile_distributions.png (selection distributions)")
    print("  - boxplots_by_duration.png (duration comparisons)")
    print("  - frames_vs_coverage.png (frame count analysis)")
    print("  - comparison_*.png (category comparisons)")
    
    print("\n📋 Tables (CSV & HTML):")
    print("  - table_global_overall_stats.csv/html (global statistics)")
    print("  - table_global_percentile_stats.csv (percentile breakdowns)")
    print("  - table_global_score_distribution.csv (score ranges)")
    print("  - comparison_table.csv (category comparison)")
    print("  - comparison_pivot.csv/html (pivot view)")
    print("  - table_*.csv/html (per-category details)")
    
    print("\n📄 Raw Data:")
    print("  - detailed_results.json (complete analysis data)")
    print("  - global_frame_stats.json (global statistics)")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    if 'overall' in summaries and summaries['overall'] and 'percentile_coverage' in summaries['overall']:
        top_95_coverage = summaries['overall']['percentile_coverage']['top_95%']['mean']
        top_99_coverage = summaries['overall']['percentile_coverage']['top_99%']['mean']
        
        print(f"1. Selection Quality:")
        if top_95_coverage < 5:
            print(f"   ⚠️  Only {top_95_coverage:.2f}% of selected frames are in top 95% (Expected: ~5%)")
            print(f"   → Selection is performing at or below random")
        elif top_95_coverage < 10:
            print(f"   ✓  {top_95_coverage:.2f}% of selected frames are in top 95%")
            print(f"   → Slightly better than random selection")
        else:
            print(f"   ✅ {top_95_coverage:.2f}% of selected frames are in top 95%")
            print(f"   → Significantly better than random!")
        
        print(f"\n2. High-Value Frame Capture:")
        print(f"   Top 95%: {top_95_coverage:.2f}% of selections")
        print(f"   Top 99%: {top_99_coverage:.2f}% of selections")
        
        if top_99_coverage > 2:
            print(f"   ✅ Good at capturing highest-value frames")
        else:
            print(f"   ⚠️  Missing many highest-value frames")
    
    print(f"\n3. Dataset Coverage:")
    print(f"   Total frames available: {global_stats['total_frames']:,}")
    if 'overall' in summaries and summaries['overall']:
        total_selected = summaries['overall']['count'] * summaries['overall']['avg_num_frames']
        selection_rate = (total_selected / global_stats['total_frames'] * 100)
        print(f"   Total frames selected: ~{total_selected:,.0f}")
        print(f"   Selection rate: {selection_rate:.2f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()