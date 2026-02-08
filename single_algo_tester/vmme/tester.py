import json
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
import warnings
import gc  # Garbage collection
import psutil  # Monitor memory
warnings.filterwarnings('ignore')

class MemorySafeMultiAlgorithmVisualizer:
    def __init__(self, metadata_path, scores_path, frames_path, 
                 algorithm_selections: Dict[str, str], 
                 output_dir="algorithm_comparison_plots",
                 n_workers=None,
                 max_memory_percent=80):  # Stop if RAM > 80%
        """
        Memory-safe visualizer with checkpoint recovery.
        
        Args:
            max_memory_percent: Stop processing if memory usage exceeds this %
        """
        print("🔄 Loading data (memory-safe mode)...")
        
        self.metadata = self._load_json(metadata_path)
        self.scores_np = [np.array(s, dtype=np.float32) for s in self._load_json(scores_path)]
        self.frames_np = [np.array(f, dtype=np.int32) for f in self._load_json(frames_path)]
        
        self.algorithms = {}
        for algo_name, path in algorithm_selections.items():
            data = self._load_json(path)
            self.algorithms[algo_name] = [np.array(d, dtype=np.int32) for d in data]
            print(f"  ✓ {algo_name}: {len(data)} videos")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        # Set matplotlib for minimal memory
        matplotlib.use('Agg')
        plt.ioff()  # Turn off interactive mode
        sns.set_style("whitegrid")
        
        # Optimized settings for lower memory
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'agg.path.chunksize': 10000,  # Reduce memory for large plots
        })
        
        self.algorithm_colors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
        ]
        self.algorithm_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', '+']
        
        # Adaptive workers based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        max_workers = max(1, min(cpu_count() - 1, int(available_memory_gb / 2)))
        self.n_workers = n_workers or max_workers
        self.max_memory_percent = max_memory_percent
        
        print(f"⚡ Workers: {self.n_workers} | Max RAM: {max_memory_percent}%")
        print(f"💾 Available RAM: {available_memory_gb:.1f} GB\n")
    
    @staticmethod
    def _load_json(path):
        """Load JSON efficiently."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_checkpoint(self, completed_ids):
        """Save progress checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({'completed': completed_ids}, f)
    
    def _load_checkpoint(self):
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f).get('completed', []))
        return set()
    
    def _check_memory(self):
        """Check if memory usage is safe."""
        mem = psutil.virtual_memory()
        if mem.percent > self.max_memory_percent:
            print(f"\n⚠️  WARNING: Memory usage at {mem.percent:.1f}%")
            print("   Forcing garbage collection...")
            gc.collect()
            return False
        return True
    
    def _get_video_data(self, doc_id):
        """Get all data for a video."""
        if doc_id >= len(self.metadata):
            return None
        
        return {
            'meta': self.metadata[doc_id],
            'frames': self.frames_np[doc_id],
            'scores': self.scores_np[doc_id],
            'selections': {
                algo_name: algo_data[doc_id] if doc_id < len(algo_data) else np.array([], dtype=np.int32)
                for algo_name, algo_data in self.algorithms.items()
            }
        }
    
    def _get_scores_vectorized(self, all_frames, all_scores, selected_frames):
        """Vectorized score lookup."""
        if len(selected_frames) == 0:
            return np.array([], dtype=np.float32)
        
        indices = np.searchsorted(all_frames, selected_frames)
        indices = np.clip(indices, 0, len(all_frames) - 1)
        
        left_indices = np.maximum(0, indices - 1)
        left_diff = np.abs(all_frames[left_indices] - selected_frames)
        right_diff = np.abs(all_frames[indices] - selected_frames)
        
        final_indices = np.where(left_diff < right_diff, left_indices, indices)
        return all_scores[final_indices]
    
    def _find_focus_region_fast(self, all_frames, *algorithm_selections):
        """Fast focus region calculation."""
        valid_sels = [s for s in algorithm_selections if len(s) > 0]
        if not valid_sels:
            return float(all_frames[0]), float(all_frames[-1])
        
        all_selected = np.concatenate(valid_sels)
        q1, q3 = np.percentile(all_selected, [25, 75])
        iqr = q3 - q1
        
        min_f, max_f = float(all_frames[0]), float(all_frames[-1])
        total_range = max_f - min_f
        
        focus_min = max(min_f, q1 - 1.5 * iqr)
        focus_max = min(max_f, q3 + 1.5 * iqr)
        
        if focus_max - focus_min < total_range * 0.3:
            center = (focus_min + focus_max) / 2
            focus_min = max(min_f, center - total_range * 0.2)
            focus_max = min(max_f, center + total_range * 0.2)
        
        return focus_min, focus_max
    
    def _plot_single_video(self, doc_id, show_top_k=True, top_k=10, dpi=200):
        """
        Memory-optimized plotting for a single video.
        
        Args:
            dpi: Lower DPI = faster + less memory (200 instead of 300)
        """
        try:
            data = self._get_video_data(doc_id)
            if data is None:
                return (doc_id, False, "Invalid doc_id")
            
            meta = data['meta']
            frames = data['frames']
            scores = data['scores']
            selections = data['selections']
            
            if all(len(sel) == 0 for sel in selections.values()):
                return (doc_id, False, "No selections")
            
            # Prepare algorithm data
            algorithm_data = {}
            all_sels = []
            for algo_name, sel_frames in selections.items():
                sel_scores = self._get_scores_vectorized(frames, scores, sel_frames)
                algorithm_data[algo_name] = (sel_frames, sel_scores)
                if len(sel_frames) > 0:
                    all_sels.append(sel_frames)
            
            focus_min, focus_max = self._find_focus_region_fast(frames, *all_sels)
            
            focus_mask = (frames >= focus_min) & (frames <= focus_max)
            focus_frames = frames[focus_mask]
            focus_scores = scores[focus_mask]
            
            if show_top_k:
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_frames = frames[top_idx]
                top_scores = scores[top_idx]
            
            # Create figure with smaller size for memory
            num_algos = len(algorithm_data)
            fig, axes = plt.subplots(
                num_algos + 1, 1,
                figsize=(28, 7 * (num_algos + 1)),  # Smaller than 32x8
                sharex=True
            )
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            max_pts = 1500  # Reduced from 2000
            step = max(1, len(focus_frames) // max_pts)
            
            # SUBPLOT 0: ALL ALGORITHMS
            ax = axes[0]
            ax.plot(focus_frames[::step], focus_scores[::step], 'o-',
                   color='gray', alpha=0.25, markersize=2, linewidth=1,
                   label='All Frames', zorder=1)
            
            if show_top_k:
                ax.scatter(top_frames, top_scores, color='gold', s=100,
                          marker='*', edgecolors='orange', linewidths=1.5,
                          label=f'Top {top_k}', zorder=4, alpha=0.7)
            
            for idx, (algo_name, (sel_f, sel_s)) in enumerate(algorithm_data.items()):
                if len(sel_f) == 0:
                    continue
                color = self.algorithm_colors[idx % len(self.algorithm_colors)]
                marker = self.algorithm_markers[idx % len(self.algorithm_markers)]
                ax.scatter(sel_f, sel_s, color=color, s=120, marker=marker,
                          edgecolors='black', linewidths=1.5,
                          label=f'{algo_name} [{len(sel_f)}]',
                          zorder=5+idx, alpha=0.75)
            
            ax.set_ylabel('Score', fontsize=14, fontweight='bold')
            ax.set_title('ALL ALGORITHMS OVERLAY', fontsize=16, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=11, framealpha=0.95, ncol=3)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            x_pad = (focus_max - focus_min) * 0.02
            ax.set_xlim(focus_min - x_pad, focus_max + x_pad)
            
            # INDIVIDUAL ALGORITHM SUBPLOTS
            for idx, (algo_name, (sel_f, sel_s)) in enumerate(algorithm_data.items(), 1):
                ax = axes[idx]
                color = self.algorithm_colors[(idx-1) % len(self.algorithm_colors)]
                marker = self.algorithm_markers[(idx-1) % len(self.algorithm_markers)]
                
                ax.plot(focus_frames[::step], focus_scores[::step], 'o-',
                       color='lightgray', alpha=0.4, markersize=3, linewidth=1,
                       label='All Frames', zorder=1)
                
                if show_top_k:
                    ax.scatter(top_frames, top_scores, color='gold', s=80,
                              marker='*', edgecolors='orange', linewidths=1.2,
                              label=f'Top {top_k}', zorder=3, alpha=0.6)
                
                if len(sel_f) > 0:
                    ax.scatter(sel_f, sel_s, color=color, s=200, marker=marker,
                              edgecolors='black', linewidths=2.5,
                              label=algo_name, zorder=6, alpha=0.9)
                    
                    in_focus_mask = (sel_f >= focus_min) & (sel_f <= focus_max)
                    for f, s in zip(sel_f[in_focus_mask], sel_s[in_focus_mask]):
                        ax.text(f, s, f'{int(f)}', fontsize=8,
                               ha='center', va='bottom', fontweight='bold',
                               color=color, zorder=10, alpha=0.95,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                        alpha=0.7, edgecolor=color, linewidth=1))
                
                left_out = sel_f[sel_f < focus_min]
                right_out = sel_f[sel_f > focus_max]
                in_focus = sel_f[(sel_f >= focus_min) & (sel_f <= focus_max)]
                
                title = f'{algo_name} | Total: {len(sel_f)} '
                title += f'(Focus: {len(in_focus)} | Left: {len(left_out)} | Right: {len(right_out)})'
                ax.set_title(title, fontsize=14, fontweight='bold', pad=10,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                                     alpha=0.2, edgecolor=color, linewidth=2))
                
                ax.set_ylabel('Score', fontsize=13, fontweight='bold')
                ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                if len(left_out) > 0:
                    l_str = ', '.join([str(int(f)) for f in left_out[:10]])
                    if len(left_out) > 10:
                        l_str += f'... +{len(left_out)-10}'
                    ax.text(0.01, 0.98, f"◄ LEFT: {l_str}",
                           transform=ax.transAxes, fontsize=8, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                                    alpha=0.9, edgecolor='black', linewidth=1.5),
                           family='monospace')
                
                if len(right_out) > 0:
                    r_str = ', '.join([str(int(f)) for f in right_out[:10]])
                    if len(right_out) > 10:
                        r_str += f'... +{len(right_out)-10}'
                    ax.text(0.99, 0.98, f"RIGHT ►: {r_str}",
                           transform=ax.transAxes, fontsize=8, va='top', ha='right',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                                    alpha=0.9, edgecolor='black', linewidth=1.5),
                           family='monospace')
            
            axes[-1].set_xlabel('Frame Index', fontsize=14, fontweight='bold')
            
            video_id = meta.get('video_id', 'N/A')
            question_id = meta.get('question_id', 'N/A')
            duration = meta.get('duration', 'N/A')
            domain = meta.get('domain', 'N/A')
            task_type = meta.get('task_type', 'N/A')
            
            suptitle = f'Multi-Algorithm Comparison (Focus: {int(focus_min)}-{int(focus_max)})\n'
            suptitle += f'Doc: {doc_id:04d} | Video: {video_id} | Q: {question_id} | '
            suptitle += f'Dur: {duration} | Domain: {domain} | Task: {task_type}'
            fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            filename = f"doc_{doc_id:04d}_{question_id}_comparison.png"
            output_path = self.output_dir / filename
            
            # Save with lower DPI for speed
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            
            # CRITICAL: Close figure and free memory
            plt.close(fig)
            del fig, axes
            gc.collect()  # Force garbage collection
            
            return (doc_id, True, filename)
            
        except Exception as e:
            plt.close('all')  # Close any open figures
            gc.collect()
            return (doc_id, False, str(e))
    
    def generate_all_plots(self, video_ids=None, max_videos=None, 
                          show_top_k=True, top_k=10, parallel=True,
                          dpi=200, batch_size=10, resume=True):
        """
        Memory-safe generation with checkpointing and batching.
        
        Args:
            video_ids: Specific video IDs to process
            max_videos: Max number of videos
            show_top_k: Show top-K frames
            top_k: Number of top frames
            parallel: Use parallel processing
            dpi: Image quality (200=fast, 300=high quality)
            batch_size: Process in batches to manage memory
            resume: Resume from checkpoint if available
        """
        print("=" * 80)
        print("🚀 MEMORY-SAFE MULTI-ALGORITHM VISUALIZATION")
        print("=" * 80)
        print(f"📁 Output: {self.output_dir.absolute()}")
        print(f"🔧 Algorithms: {', '.join(self.algorithms.keys())}")
        print(f"⚡ Parallel: {'Yes' if parallel else 'No'} ({self.n_workers} workers)")
        print(f"🎨 DPI: {dpi} | Batch size: {batch_size}")
        print(f"💾 Memory limit: {self.max_memory_percent}%")
        print("=" * 80 + "\n")
        
        # Determine videos to process
        total = len(self.metadata)
        if video_ids is not None:
            to_process = [v for v in video_ids if v < total]
        elif max_videos is not None:
            to_process = list(range(min(max_videos, total)))
        else:
            to_process = list(range(total))
        
        # Load checkpoint
        completed = self._load_checkpoint() if resume else set()
        if completed:
            print(f"📋 Found checkpoint: {len(completed)} videos already completed")
            to_process = [v for v in to_process if v not in completed]
            print(f"📊 Remaining: {len(to_process)} videos\n")
        else:
            print(f"📊 Processing {len(to_process)} video(s)...\n")
        
        if not to_process:
            print("✅ All videos already processed!")
            return []
        
        # Process in batches
        all_results = []
        total_batches = (len(to_process) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(to_process), batch_size):
            batch = to_process[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            print(f"\n{'='*60}")
            print(f"📦 BATCH {batch_num}/{total_batches} (Videos {batch[0]}-{batch[-1]})")
            print(f"{'='*60}")
            
            # Check memory before batch
            mem = psutil.virtual_memory()
            print(f"💾 Memory before batch: {mem.percent:.1f}% used ({mem.available / (1024**3):.1f} GB free)")
            
            if mem.percent > self.max_memory_percent:
                print(f"⚠️  WARNING: Memory too high! Running garbage collection...")
                gc.collect()
                mem = psutil.virtual_memory()
                if mem.percent > self.max_memory_percent:
                    print(f"❌ ERROR: Still at {mem.percent:.1f}% after GC. Stopping.")
                    print(f"   Try: Lower batch_size, use parallel=False, or increase RAM")
                    break
            
            # Process batch
            if parallel and len(batch) > 1:
                # PARALLEL
                with Pool(processes=min(self.n_workers, len(batch))) as pool:
                    args = [(vid, show_top_k, top_k, dpi) for vid in batch]
                    batch_results = pool.starmap(self._plot_single_video, args)
            else:
                # SEQUENTIAL
                batch_results = []
                for i, vid in enumerate(batch, 1):
                    print(f"  [{i}/{len(batch)}] ", end="", flush=True)
                    result = self._plot_single_video(vid, show_top_k, top_k, dpi)
                    batch_results.append(result)
                    
                    if result[1]:
                        print(f"✓ {result[2]}")
                    else:
                        print(f"✗ doc_{vid:04d}: {result[2]}")
                    
                    # Check memory after each video in sequential mode
                    if not self._check_memory():
                        print("   Pausing for memory cleanup...")
                        gc.collect()
            
            all_results.extend(batch_results)
            
            # Update checkpoint after each batch
            batch_completed = [r[0] for r in batch_results if r[1]]
            completed.update(batch_completed)
            self._save_checkpoint(list(completed))
            
            # Memory cleanup after batch
            gc.collect()
            mem_after = psutil.virtual_memory()
            print(f"💾 Memory after batch: {mem_after.percent:.1f}% used")
            print(f"✓ Batch {batch_num} complete: {len(batch_completed)}/{len(batch)} successful")
            
            # Safety pause between batches if memory is high
            if mem_after.percent > 70:
                print("⏸️  High memory usage - extra cleanup...")
                plt.close('all')
                gc.collect()
        
        # Final summary
        successes = sum(1 for r in all_results if r[1])
        failures = len(all_results) - successes
        
        print(f"\n{'=' * 80}")
        print(f"✅ GENERATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"✓ Successfully processed: {successes}/{len(all_results)} plots")
        print(f"✓ Total completed (including previous): {len(completed)}")
        if failures > 0:
            print(f"✗ Failed: {failures}")
            print("\nFailed videos:")
            for doc_id, success, error in all_results:
                if not success:
                    print(f"  - doc_{doc_id:04d}: {error}")
        print(f"📁 Output location: {self.output_dir.absolute()}")
        print(f"💾 Final memory usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"{'=' * 80}\n")
        
        return all_results
    
    def generate_summary_statistics(self):
        """Generate summary statistics comparing all algorithms."""
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        
        for algo_name, algo_selections in self.algorithms.items():
            frame_counts = [len(sel) for sel in algo_selections if len(sel) > 0]
            
            if len(frame_counts) == 0:
                print(f"\n{algo_name}:")
                print(f"  No selections found")
                continue
            
            print(f"\n{algo_name}:")
            print(f"  Videos with selections: {len(frame_counts)}/{len(algo_selections)}")
            print(f"  Avg frames per video: {np.mean(frame_counts):.2f}")
            print(f"  Min frames: {np.min(frame_counts)}")
            print(f"  Max frames: {np.max(frame_counts)}")
            print(f"  Median frames: {np.median(frame_counts):.2f}")
            print(f"  Std dev: {np.std(frame_counts):.2f}")
            print(f"  Total frames selected: {np.sum(frame_counts)}")
        
        print("\n" + "=" * 80 + "\n")
    
    def compare_algorithm_overlap(self, doc_id):
        """Analyze frame overlap between algorithms."""
        data = self._get_video_data(doc_id)
        if data is None:
            return {}
        
        selections = data['selections']
        algo_names = list(selections.keys())
        overlap_matrix = {}
        
        for i, algo1 in enumerate(algo_names):
            for j, algo2 in enumerate(algo_names):
                if i >= j:
                    continue
                
                frames1 = set(selections[algo1])
                frames2 = set(selections[algo2])
                
                if len(frames1) == 0 or len(frames2) == 0:
                    overlap_matrix[(algo1, algo2)] = {
                        'intersection': 0,
                        'union': 0,
                        'jaccard': 0.0,
                        'overlap_pct': 0.0
                    }
                    continue
                
                intersection = frames1.intersection(frames2)
                union = frames1.union(frames2)
                
                overlap_matrix[(algo1, algo2)] = {
                    'intersection': len(intersection),
                    'union': len(union),
                    'jaccard': len(intersection) / len(union) if len(union) > 0 else 0.0,
                    'overlap_pct': len(intersection) / min(len(frames1), len(frames2)) * 100
                }
        
        return overlap_matrix
    
    def generate_overlap_report(self, doc_ids=None, max_videos=10):
        """Generate overlap analysis report."""
        print("\n" + "=" * 80)
        print("🔍 ALGORITHM OVERLAP ANALYSIS")
        print("=" * 80)
        
        if doc_ids is None:
            doc_ids = list(range(min(max_videos, len(self.metadata))))
        
        for doc_id in doc_ids:
            try:
                data = self._get_video_data(doc_id)
                if data is None:
                    print(f"✗ doc_{doc_id:04d}: Invalid ID")
                    continue
                
                meta = data['meta']
                video_id = meta.get('video_id', 'N/A')
                question_id = meta.get('question_id', 'N/A')
                
                print(f"\n📹 Doc {doc_id:04d} | Video: {video_id} | Q: {question_id}")
                print("-" * 80)
                
                overlap_matrix = self.compare_algorithm_overlap(doc_id)
                
                if not overlap_matrix:
                    print("  No overlap data available")
                    continue
                
                for (algo1, algo2), stats in overlap_matrix.items():
                    print(f"  {algo1} ↔ {algo2}:")
                    print(f"    Common frames: {stats['intersection']}")
                    print(f"    Jaccard similarity: {stats['jaccard']:.2%}")
                    print(f"    Overlap percentage: {stats['overlap_pct']:.1f}%")
                
            except Exception as e:
                print(f"  ✗ Error analyzing doc_id {doc_id}: {e}")
        
        print("\n" + "=" * 80 + "\n")
    
    def clear_checkpoint(self):
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("✓ Checkpoint cleared")


# ============================================================================
# MAIN EXECUTION WITH SAFETY FEATURES
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    METADATA_PATH = "metadata_LV.json"
    SCORES_PATH = "scores_LV.json"
    FRAMES_PATH = "frames_LV_clip.json"
    
    # ALGORITHM_SELECTIONS = {
    #     "AKS": "selected_longvideobench__clip_frames_k32_aks.json",
    #      "PPTMAS" : "PPTMAS/selected_pptmas_longvideobench_clip_k32_a0.4_b0.15_g0.6_opt.json",
    #      "halfLife": "PPTMAS/selected_pptmas_longvideobench_half_life_clip_k32_a0.4_b0.15_g0.6_opt.json",
    #      "Fixed" : "PPTMAS/selected_pptmas_longvideobench_clip_k32_w2.0_d0.1_opt.json",
    #      "param_free": "selected_tmas_longvideobench_clip_k32_auto_half_life_curvseco_normmax_clip95_hybrid_cov1.0_opt.json",
    #      "recom": "selected_pptmas_longvideobench_clip_k32_w2.0_d0.1_opt_recom.json",
        
        
    # }
    

    # ALGORITHM_SELECTIONS = {
    # "AKS": "Selected_frames_tmas_TmasParamFree/selected_videomme_clip_frames_k32_aks.json",
    # "DBFP": "Selected_frames_tmas_TmasParamFree/selected_dbfp_tmas_videomme_clip_k32_auto_hybrid_cov1.73_half_life_alpha0.85_temporal_iter0.json",
    
    # # Budget-based variants
    # "budget_grad": "Selected_frames_tmas_TmasParamFree/selected_tmas_videomme_clip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.0_opt.json",
    # "budget_lapl": "Selected_frames_tmas_TmasParamFree/selected_tmas_videomme_clip_k32_auto_budget_based_curvlapl_normmax_clip95_hybrid_cov1.0_opt.json",
    # "budget_seco": "Selected_frames_tmas_TmasParamFree/selected_tmas_videomme_clip_k32_auto_budget_based_curvseco_normmax_clip95_hybrid_cov1.0_opt.json",
    
    # # Half-life variants (param-free)
    # "halflife_grad": "Selected_frames_tmas_TmasParamFree/selected_tmas_videomme_clip_k32_auto_half_life_curvgrad_normmax_clip95_hybrid_cov1.0_opt.json",
    # "halflife_lapl": "Selected_frames_tmas_TmasParamFree/selected_tmas_videomme_clip_k32_auto_half_life_curvlapl_normmax_clip95_hybrid_cov1.0_opt.json",
    # "halflife_seco": "Selected_frames_tmas_TmasParamFree/selected_tmas_videomme_clip_k32_auto_half_life_curvseco_normmax_clip95_hybrid_cov1.0_opt.json",
    
    # }

#     ALGORITHM_SELECTIONS = {
#     "AKS": "selected_frames_LV/selected_longvideobench_clip_frames_k32_aks.json",
    
#     # Budget-based variants
#     "budget_grad": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.0_opt.json",
#     "budget_lapl": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_budget_based_curvlapl_normmax_clip95_hybrid_cov1.0_opt.json",
#     "budget_seco": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_budget_based_curvseco_normmax_clip95_hybrid_cov1.0_opt.json",
#     "budget_base": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_budget_based_hybrid_cov1.0_opt.json",
    
#     # Half-life variants (param-free)
#     "halflife_grad": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_half_life_curvgrad_normmax_clip95_hybrid_cov1.0_opt.json",
#     "halflife_lapl": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_half_life_curvlapl_normmax_clip95_hybrid_cov1.0_opt.json",
#     "halflife_seco": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_half_life_curvseco_normmax_clip95_hybrid_cov1.0_opt.json",
#     "halflife_base": "selected_frames_LV/selected_tmas_longvideobench_clip_k32_auto_half_life_hybrid_cov1.0_opt.json",
    
# }

    ALGORITHM_SELECTIONS = {
    "AKS": "/home/train01/aks/Frame_selection/THESIS/Thesis_aks/longvideobench/selected_longvideobench_blip_aks32_ratio1.json",
    "TMAS_budget" : "/home/train01/aks/Frame_selection/THESIS/longvideobench/selected_tmas_longvideobench_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_opt.json",
    "TMAS_half_life" : "/home/train01/aks/Frame_selection/THESIS/longvideobench/selected_tmas_longvideobench_blip_k32_auto_half_life_curvgrad_normmax_clip95_hybrid_cov1.73_opt.json" 
}

    OUTPUT_DIR = "LV_curative_test"
    
    # ========================================================================
    # INITIALIZE WITH MEMORY SAFETY
    # ========================================================================
    
    try:
        visualizer = MemorySafeMultiAlgorithmVisualizer(
            metadata_path=METADATA_PATH,
            scores_path=SCORES_PATH,
            frames_path=FRAMES_PATH,
            algorithm_selections=ALGORITHM_SELECTIONS,
            output_dir=OUTPUT_DIR,
            n_workers=32,  # Auto-detect based on RAM
            max_memory_percent=95  # Stop if RAM > 80%
        )
        
        # ====================================================================
        # SAFE GENERATION OPTIONS
        # ====================================================================
        
        # OPTION 1: Full processing with checkpointing (RECOMMENDED)
        # If it crashes, just run again - it will resume from checkpoint!
        visualizer.generate_all_plots(
            parallel=True,      # Use parallel processing
            dpi=200,            # Lower DPI = faster (use 300 for final output)
            batch_size=1000,      # Process 10 videos at a time
            resume=True         # Resume from checkpoint if crashed
        )
        
        # OPTION 2: Test with small batch first
        # visualizer.generate_all_plots(
        #     max_videos=5,
        #     parallel=False,  # Sequential for testing
        #     dpi=150,         # Even lower for testing
        #     batch_size=5,
        #     resume=False
        # )
        
        # OPTION 3: Process specific videos
        # visualizer.generate_all_plots(
        #     video_ids=[0, 1, 2, 3, 4],
        #     parallel=True,
        #     dpi=200,
        #     batch_size=5
        # )
        
        # OPTION 4: High quality final run (after testing)
        # visualizer.generate_all_plots(
        #     parallel=True,
        #     dpi=300,         # High quality
        #     batch_size=5,    # Smaller batches for large images
        #     resume=True
        # )
        
        # ====================================================================
        # GENERATE STATISTICS
        # ====================================================================
        
        visualizer.generate_summary_statistics()
        # visualizer.generate_overlap_report(max_videos=10)
        
        print("\n" + "=" * 80)
        print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("✓ Progress saved to checkpoint - run again to resume")
    except MemoryError:
        print("\n\n❌ OUT OF MEMORY ERROR")
        print("💡 Solutions:")
        print("   1. Reduce batch_size (try 5 or 3)")
        print("   2. Lower DPI (try 150)")
        print("   3. Use parallel=False")
        print("   4. Close other applications")
        print("   5. Process fewer videos at once")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print("✓ Progress saved to checkpoint")
        import traceback
        traceback.print_exc()