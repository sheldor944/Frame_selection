import json
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import warnings
import gc  # Garbage collection
import psutil  # Monitor memory
from collections import defaultdict
warnings.filterwarnings('ignore')


class VideoMetadataOrganizer:
    """
    Organizes video metadata by question categories and manages folder structure.
    """
    
    def __init__(self, video_metadata_path: str):
        """
        Initialize with the new video metadata JSON.
        
        Args:
            video_metadata_path: Path to JSON file with video metadata containing question_category
        """
        self.video_metadata = self._load_json(video_metadata_path)
        self.category_mapping = self._build_category_mapping()
        self.categories = list(self.category_mapping.keys())
        
        print(f"📂 Loaded {len(self.video_metadata)} video entries")
        print(f"📁 Found {len(self.categories)} unique categories:")
        for cat, videos in self.category_mapping.items():
            print(f"   • {cat}: {len(videos)} videos")
    
    @staticmethod
    def _load_json(path: str):
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_category_mapping(self) -> Dict[str, List[dict]]:
        """
        Build a mapping of question_category -> list of video entries.
        """
        category_map = defaultdict(list)
        
        for idx, entry in enumerate(self.video_metadata):
            # Get category - use 'unknown' if not present
            category = entry.get('question_category', 'unknown')
            
            # Also store the original index for reference
            entry['_original_idx'] = idx
            category_map[category].append(entry)
        
        return dict(category_map)
    
    def get_videos_by_category(self, category: str) -> List[dict]:
        """Get all videos in a specific category."""
        return self.category_mapping.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories."""
        return self.categories
    
    def get_video_by_id(self, video_id: str) -> Optional[dict]:
        """Find a video entry by its video_id."""
        for entry in self.video_metadata:
            if entry.get('video_id') == video_id:
                return entry
        return None
    
    def get_entry_by_doc_id(self, doc_id: str) -> Optional[dict]:
        """Find a video entry by its id field (e.g., '86CxyhFV9MI_0')."""
        for entry in self.video_metadata:
            if entry.get('id') == doc_id:
                return entry
        return None


class MemorySafeMultiAlgorithmVisualizer:
    def __init__(self, metadata_path: str, scores_path: str, frames_path: str, 
                 algorithm_selections: Dict[str, str],
                 video_metadata_path: str,  # NEW: Path to video metadata with categories
                 output_dir: str = "algorithm_comparison_plots",
                 n_workers: Optional[int] = None,
                 max_memory_percent: int = 80):
        """
        Memory-safe visualizer with category-based folder organization.
        
        Args:
            metadata_path: Path to original metadata
            scores_path: Path to scores JSON
            frames_path: Path to frames JSON
            algorithm_selections: Dict mapping algorithm names to their selection JSON paths
            video_metadata_path: Path to NEW JSON with video metadata including question_category
            output_dir: Parent output directory
            n_workers: Number of parallel workers
            max_memory_percent: Stop if RAM exceeds this percentage
        """
        print("=" * 80)
        print("🔄 INITIALIZING MEMORY-SAFE VISUALIZER WITH CATEGORY ORGANIZATION")
        print("=" * 80)
        
        # Load original data
        print("\n📥 Loading original data...")
        self.metadata = self._load_json(metadata_path)
        self.scores_np = [np.array(s, dtype=np.float32) for s in self._load_json(scores_path)]
        self.frames_np = [np.array(f, dtype=np.int32) for f in self._load_json(frames_path)]
        print(f"   ✓ Metadata: {len(self.metadata)} entries")
        print(f"   ✓ Scores: {len(self.scores_np)} entries")
        print(f"   ✓ Frames: {len(self.frames_np)} entries")
        
        # Load algorithms
        print("\n📥 Loading algorithm selections...")
        self.algorithms = {}
        for algo_name, path in algorithm_selections.items():
            data = self._load_json(path)
            self.algorithms[algo_name] = [np.array(d, dtype=np.int32) for d in data]
            print(f"   ✓ {algo_name}: {len(data)} videos")
        
        # Initialize video metadata organizer
        print("\n📥 Loading video metadata with categories...")
        self.video_organizer = VideoMetadataOrganizer(video_metadata_path)
        
        # Create parent output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create category subdirectories
        print("\n📁 Creating category folders...")
        self.category_dirs = {}
        for category in self.video_organizer.get_all_categories():
            # Sanitize category name for folder (replace special chars)
            safe_category = self._sanitize_folder_name(category)
            cat_dir = self.output_dir / safe_category
            cat_dir.mkdir(exist_ok=True, parents=True)
            self.category_dirs[category] = cat_dir
            print(f"   ✓ Created: {cat_dir}")
        
        # Create 'unknown' folder for videos without category
        unknown_dir = self.output_dir / "unknown_category"
        unknown_dir.mkdir(exist_ok=True, parents=True)
        self.category_dirs['unknown'] = unknown_dir
        
        # Build doc_id to category mapping for fast lookup
        self._build_doc_id_to_category_map()
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        # Set matplotlib for minimal memory
        matplotlib.use('Agg')
        plt.ioff()
        sns.set_style("whitegrid")
        
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'agg.path.chunksize': 10000,
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
        
        print(f"\n⚡ Workers: {self.n_workers} | Max RAM: {max_memory_percent}%")
        print(f"💾 Available RAM: {available_memory_gb:.1f} GB")
        print("=" * 80 + "\n")
    
    @staticmethod
    def _load_json(path: str):
        """Load JSON efficiently."""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        """Sanitize a string to be used as a folder name."""
        # Replace problematic characters
        replacements = {
            '/': '_',
            '\\': '_',
            ':': '_',
            '*': '_',
            '?': '_',
            '"': '_',
            '<': '_',
            '>': '_',
            '|': '_',
            ' ': '_',
        }
        result = name
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
    
    def _build_doc_id_to_category_map(self):
        """
        Build a mapping from doc_id (index in metadata) to question_category.
        
        This handles the case where the original metadata might use different
        indexing than the new video metadata.
        """
        self.doc_id_to_category = {}
        self.doc_id_to_video_meta = {}
        
        # Try to match by video_id or other common fields
        for doc_id, meta in enumerate(self.metadata):
            # Get identifiers from original metadata
            video_id = meta.get('video_id', '')
            question_id = meta.get('question_id', meta.get('id', ''))
            
            # Try to find matching entry in video metadata
            matched_entry = None
            
            # Try matching by 'id' field (e.g., '86CxyhFV9MI_0')
            if question_id:
                matched_entry = self.video_organizer.get_entry_by_doc_id(question_id)
            
            # If not found, try matching by video_id
            if matched_entry is None and video_id:
                # This might match multiple entries, take the first
                for entry in self.video_organizer.video_metadata:
                    if entry.get('video_id') == video_id:
                        matched_entry = entry
                        break
            
            # If still not found, try matching by index
            if matched_entry is None and doc_id < len(self.video_organizer.video_metadata):
                matched_entry = self.video_organizer.video_metadata[doc_id]
            
            if matched_entry:
                category = matched_entry.get('question_category', 'unknown')
                self.doc_id_to_category[doc_id] = category
                self.doc_id_to_video_meta[doc_id] = matched_entry
            else:
                self.doc_id_to_category[doc_id] = 'unknown'
                self.doc_id_to_video_meta[doc_id] = None
        
        # Print mapping summary
        category_counts = defaultdict(int)
        for cat in self.doc_id_to_category.values():
            category_counts[cat] += 1
        
        print("\n📊 Doc ID to Category Mapping:")
        for cat, count in sorted(category_counts.items()):
            print(f"   • {cat}: {count} documents")
    
    def _get_output_path_for_doc(self, doc_id: int, filename: str) -> Path:
        """
        Get the appropriate output path for a document based on its category.
        """
        category = self.doc_id_to_category.get(doc_id, 'unknown')
        category_dir = self.category_dirs.get(category, self.category_dirs['unknown'])
        return category_dir / filename
    
    def _save_checkpoint(self, completed_ids: List[int]):
        """Save progress checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({'completed': completed_ids}, f)
    
    def _load_checkpoint(self) -> set:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f).get('completed', []))
        return set()
    
    def _check_memory(self) -> bool:
        """Check if memory usage is safe."""
        mem = psutil.virtual_memory()
        if mem.percent > self.max_memory_percent:
            print(f"\n⚠️  WARNING: Memory usage at {mem.percent:.1f}%")
            print("   Forcing garbage collection...")
            gc.collect()
            return False
        return True
    
    def _get_video_data(self, doc_id: int) -> Optional[dict]:
        """Get all data for a video."""
        if doc_id >= len(self.metadata):
            return None
        
        return {
            'meta': self.metadata[doc_id],
            'video_meta': self.doc_id_to_video_meta.get(doc_id),
            'category': self.doc_id_to_category.get(doc_id, 'unknown'),
            'frames': self.frames_np[doc_id],
            'scores': self.scores_np[doc_id],
            'selections': {
                algo_name: algo_data[doc_id] if doc_id < len(algo_data) else np.array([], dtype=np.int32)
                for algo_name, algo_data in self.algorithms.items()
            }
        }
    
    def _get_scores_vectorized(self, all_frames: np.ndarray, all_scores: np.ndarray, 
                               selected_frames: np.ndarray) -> np.ndarray:
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
    
    def _find_focus_region_fast(self, all_frames: np.ndarray, 
                                *algorithm_selections) -> Tuple[float, float]:
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
    
    def _plot_single_video(self, doc_id: int, show_top_k: bool = True, 
                           top_k: int = 10, dpi: int = 200) -> Tuple[int, bool, str]:
        """
        Memory-optimized plotting for a single video with category-based output.
        """
        try:
            data = self._get_video_data(doc_id)
            if data is None:
                return (doc_id, False, "Invalid doc_id")
            
            meta = data['meta']
            video_meta = data['video_meta']
            category = data['category']
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
            
            # Create figure
            num_algos = len(algorithm_data)
            fig, axes = plt.subplots(
                num_algos + 1, 1,
                figsize=(28, 7 * (num_algos + 1)),
                sharex=True
            )
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            max_pts = 1500
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
            ax.set_title(f'ALL ALGORITHMS OVERLAY | Category: {category}', 
                        fontsize=16, fontweight='bold', pad=15)
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
            
            # Get metadata for title
            video_id = meta.get('video_id', 'N/A')
            question_id = meta.get('question_id', meta.get('id', 'N/A'))
            duration = meta.get('duration', 'N/A')
            domain = meta.get('domain', 'N/A')
            task_type = meta.get('task_type', 'N/A')
            
            # Add info from video_meta if available
            if video_meta:
                topic_cat = video_meta.get('topic_category', 'N/A')
                level = video_meta.get('level', 'N/A')
            else:
                topic_cat = 'N/A'
                level = 'N/A'
            
            suptitle = f'Multi-Algorithm Comparison (Focus: {int(focus_min)}-{int(focus_max)})\n'
            suptitle += f'Doc: {doc_id:04d} | Video: {video_id} | Q: {question_id}\n'
            suptitle += f'Category: {category} | Topic: {topic_cat} | Level: {level}'
            fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Generate filename and get category-based output path
            filename = f"doc_{doc_id:04d}_{self._sanitize_folder_name(str(question_id))}_comparison.png"
            output_path = self._get_output_path_for_doc(doc_id, filename)
            
            # Save
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            
            # Cleanup
            plt.close(fig)
            del fig, axes
            gc.collect()
            
            return (doc_id, True, str(output_path.relative_to(self.output_dir)))
            
        except Exception as e:
            plt.close('all')
            gc.collect()
            return (doc_id, False, str(e))
    
    def generate_all_plots(self, video_ids: Optional[List[int]] = None, 
                          max_videos: Optional[int] = None,
                          categories: Optional[List[str]] = None,  # NEW: Filter by category
                          show_top_k: bool = True, top_k: int = 10, 
                          parallel: bool = True,
                          dpi: int = 200, batch_size: int = 10, 
                          resume: bool = True) -> List[Tuple]:
        """
        Memory-safe generation with checkpointing, batching, and category filtering.
        
        Args:
            video_ids: Specific video IDs to process
            max_videos: Max number of videos
            categories: List of categories to process (None = all)
            show_top_k: Show top-K frames
            top_k: Number of top frames
            parallel: Use parallel processing
            dpi: Image quality
            batch_size: Process in batches
            resume: Resume from checkpoint
        """
        print("=" * 80)
        print("🚀 MEMORY-SAFE MULTI-ALGORITHM VISUALIZATION WITH CATEGORIES")
        print("=" * 80)
        print(f"📁 Output: {self.output_dir.absolute()}")
        print(f"🔧 Algorithms: {', '.join(self.algorithms.keys())}")
        print(f"⚡ Parallel: {'Yes' if parallel else 'No'} ({self.n_workers} workers)")
        print(f"🎨 DPI: {dpi} | Batch size: {batch_size}")
        print(f"💾 Memory limit: {self.max_memory_percent}%")
        
        # Determine videos to process
        total = len(self.metadata)
        
        if video_ids is not None:
            to_process = [v for v in video_ids if v < total]
        elif categories is not None:
            # Filter by categories
            to_process = []
            for doc_id, cat in self.doc_id_to_category.items():
                if cat in categories:
                    to_process.append(doc_id)
            print(f"\n📂 Filtering by categories: {categories}")
        elif max_videos is not None:
            to_process = list(range(min(max_videos, total)))
        else:
            to_process = list(range(total))
        
        # Print category distribution for videos to process
        cat_counts = defaultdict(int)
        for doc_id in to_process:
            cat = self.doc_id_to_category.get(doc_id, 'unknown')
            cat_counts[cat] += 1
        
        print(f"\n📊 Videos to process by category:")
        for cat, count in sorted(cat_counts.items()):
            print(f"   • {cat}: {count}")
        
        print("=" * 80 + "\n")
        
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
                    break
            
            # Process batch
            if parallel and len(batch) > 1:
                with Pool(processes=min(self.n_workers, len(batch))) as pool:
                    args = [(vid, show_top_k, top_k, dpi) for vid in batch]
                    batch_results = pool.starmap(self._plot_single_video, args)
            else:
                batch_results = []
                for i, vid in enumerate(batch, 1):
                    cat = self.doc_id_to_category.get(vid, 'unknown')
                    print(f"  [{i}/{len(batch)}] doc_{vid:04d} ({cat}): ", end="", flush=True)
                    result = self._plot_single_video(vid, show_top_k, top_k, dpi)
                    batch_results.append(result)
                    
                    if result[1]:
                        print(f"✓ {result[2]}")
                    else:
                        print(f"✗ {result[2]}")
                    
                    if not self._check_memory():
                        print("   Pausing for memory cleanup...")
                        gc.collect()
            
            all_results.extend(batch_results)
            
            # Update checkpoint
            batch_completed = [r[0] for r in batch_results if r[1]]
            completed.update(batch_completed)
            self._save_checkpoint(list(completed))
            
            # Memory cleanup
            gc.collect()
            mem_after = psutil.virtual_memory()
            print(f"💾 Memory after batch: {mem_after.percent:.1f}% used")
            print(f"✓ Batch {batch_num} complete: {len(batch_completed)}/{len(batch)} successful")
            
            if mem_after.percent > 70:
                print("⏸️  High memory usage - extra cleanup...")
                plt.close('all')
                gc.collect()
        
        # Final summary
        successes = sum(1 for r in all_results if r[1])
        failures = len(all_results) - successes
        
        # Summary by category
        success_by_cat = defaultdict(int)
        fail_by_cat = defaultdict(int)
        for doc_id, success, _ in all_results:
            cat = self.doc_id_to_category.get(doc_id, 'unknown')
            if success:
                success_by_cat[cat] += 1
            else:
                fail_by_cat[cat] += 1
        
        print(f"\n{'=' * 80}")
        print(f"✅ GENERATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"✓ Successfully processed: {successes}/{len(all_results)} plots")
        print(f"✓ Total completed (including previous): {len(completed)}")
        
        print(f"\n📊 Results by category:")
        all_cats = set(success_by_cat.keys()) | set(fail_by_cat.keys())
        for cat in sorted(all_cats):
            s = success_by_cat.get(cat, 0)
            f = fail_by_cat.get(cat, 0)
            print(f"   • {cat}: {s} success, {f} failed")
        
        if failures > 0:
            print(f"\n✗ Failed: {failures}")
            print("\nFailed videos:")
            for doc_id, success, error in all_results:
                if not success:
                    cat = self.doc_id_to_category.get(doc_id, 'unknown')
                    print(f"  - doc_{doc_id:04d} ({cat}): {error}")
        
        print(f"\n📁 Output location: {self.output_dir.absolute()}")
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
    
    def generate_category_report(self):
        """Generate a report of videos organized by category."""
        print("\n" + "=" * 80)
        print("📂 CATEGORY ORGANIZATION REPORT")
        print("=" * 80)
        
        for category in sorted(self.video_organizer.get_all_categories()):
            videos = self.video_organizer.get_videos_by_category(category)
            cat_dir = self.category_dirs.get(category, self.category_dirs['unknown'])
            
            # Count generated plots in this category
            existing_plots = list(cat_dir.glob("*.png"))
            
            print(f"\n📁 {category}:")
            print(f"   Directory: {cat_dir}")
            print(f"   Videos in metadata: {len(videos)}")
            print(f"   Plots generated: {len(existing_plots)}")
            
            if len(videos) > 0:
                # Sample video IDs
                sample_ids = [v.get('video_id', 'N/A') for v in videos[:5]]
                print(f"   Sample video IDs: {', '.join(sample_ids)}")
        
        print("\n" + "=" * 80 + "\n")
    
    def clear_checkpoint(self):
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("✓ Checkpoint cleared")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Original data paths
    METADATA_PATH = "/home/train01/aks/Frame_selection/single_algo_tester/vmme/metadata_LV.json"
    SCORES_PATH = "/home/train01/aks/Frame_selection/outscores/longvideobench/blip/scores.json"
    FRAMES_PATH = "/home/train01/aks/Frame_selection/outscores/longvideobench/blip/frames.json"
    
    # NEW: Path to video metadata with question_category
    VIDEO_METADATA_PATH = "/home/train01/aks/Frame_selection/THESIS_LV/selected_tmas_longvideobench_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_opt.json"  # Your new JSON file
    
    # Algorithm selections
    ALGORITHM_SELECTIONS = {
        "AKS": "/home/train01/aks/Frame_selection/THESIS/Thesis_aks/longvideobench/selected_longvideobench_blip_aks32_ratio1.json",
        "TMAS_budget": "/home/train01/aks/Frame_selection/THESIS/longvideobench/selected_tmas_longvideobench_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_opt.json",
        "TMAS_half_life": "/home/train01/aks/Frame_selection/THESIS/longvideobench/selected_tmas_longvideobench_blip_k32_auto_half_life_curvgrad_normmax_clip95_hybrid_cov1.73_opt.json"
    }
    
    # Parent output directory - subfolders will be created per category
    OUTPUT_DIR = "LV_by_catagory_THESIS"
    
    # ========================================================================
    # INITIALIZE
    # ========================================================================
    
    try:
        visualizer = MemorySafeMultiAlgorithmVisualizer(
            metadata_path=METADATA_PATH,
            scores_path=SCORES_PATH,
            frames_path=FRAMES_PATH,
            algorithm_selections=ALGORITHM_SELECTIONS,
            video_metadata_path=VIDEO_METADATA_PATH,  # NEW parameter
            output_dir=OUTPUT_DIR,
            n_workers=32,
            max_memory_percent=95
        )
        
        # ====================================================================
        # GENERATION OPTIONS
        # ====================================================================
        
        # Generate category report first
        visualizer.generate_category_report()
        
        # OPTION 1: Process ALL videos (organized by category)
        visualizer.generate_all_plots(
            parallel=True,
            dpi=100,
            batch_size=1000,
            resume=True
        )
        
        # OPTION 2: Process only specific categories
        # visualizer.generate_all_plots(
        #     categories=['TOS', 'TPO'],  # Only these categories
        #     parallel=True,
        #     dpi=200,
        #     batch_size=100,
        #     resume=True
        # )
        
        # OPTION 3: Test with small batch
        # visualizer.generate_all_plots(
        #     max_videos=10,
        #     parallel=False,
        #     dpi=150,
        #     batch_size=5,
        #     resume=False
        # )
        
        # Generate statistics
        visualizer.generate_summary_statistics()
        
        print("\n" + "=" * 80)
        print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("✓ Progress saved to checkpoint - run again to resume")
    except MemoryError:
        print("\n\n❌ OUT OF MEMORY ERROR")
        print("💡 Try: Lower batch_size, use parallel=False, or close other apps")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()