import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


def generate_minimal_publication_plot(
    doc_id: int,
    metadata_path: str,
    scores_path: str,
    frames_path: str,
    output_dir: str = "publication_plots",
    dpi: int = 600,
):
    """
    Generate publication-ready minimal plots with multiple variations.
    Creates both transparent and white background versions.
    
    Args:
        doc_id: Document ID to visualize
        metadata_path: Path to metadata JSON
        scores_path: Path to scores JSON
        frames_path: Path to frames JSON
        output_dir: Output directory
        dpi: Resolution (600 for publication)
    """
    
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)
    
    print("=" * 70)
    print(f"📊 GENERATING PUBLICATION-READY PLOTS FOR DOCUMENT {doc_id}")
    print("=" * 70)
    
    # Load data
    metadata = load_json(metadata_path)
    scores_data = load_json(scores_path)
    frames_data = load_json(frames_path)
    
    if doc_id >= len(metadata):
        raise ValueError(f"doc_id {doc_id} out of range (max: {len(metadata)-1})")
    
    frames = np.array(frames_data[doc_id], dtype=np.float64)
    scores = np.array(scores_data[doc_id], dtype=np.float64)
    
    # Sort by frames
    sort_idx = np.argsort(frames)
    frames = frames[sort_idx]
    scores = scores[sort_idx]
    
    print(f"   Frames: {len(frames)}")
    print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Smooth the curve
    scores_smooth = gaussian_filter1d(scores, sigma=4)
    
    # ========================================================================
    # VARIATION 1: WHITE BACKGROUND (for print/PDF)
    # ========================================================================
    
    print("\n📈 Variation 1: White Background (Print Ready)...")
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot(frames, scores_smooth, color='#1a56db', linewidth=2.0, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=13, fontweight='medium', labelpad=8)
    ax.set_ylabel('Relevance Score', fontsize=13, fontweight='medium', labelpad=8)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    ax.grid(False)
    
    # Add arrows to axes
    ax.plot(1, 0, ">", color='#333333', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=7)
    ax.plot(0, 1, "^", color='#333333', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=7)
    
    # Adjust limits for padding
    x_pad = (frames[-1] - frames[0]) * 0.02
    y_pad = (scores_smooth.max() - scores_smooth.min()) * 0.08
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    path1 = output_dir / f"doc{doc_id}_minimal_white.png"
    plt.savefig(path1, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Saved: {path1}")
    
    # ========================================================================
    # VARIATION 2: TRANSPARENT BACKGROUND (PNG with alpha)
    # ========================================================================
    
    print("📈 Variation 2: Transparent Background (PNG)...")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0.0)  # Transparent figure
    ax.set_facecolor('none')   # Transparent axes
    
    ax.plot(frames, scores_smooth, color='#1a56db', linewidth=2.0, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=13, fontweight='medium', labelpad=8, color='#222222')
    ax.set_ylabel('Relevance Score', fontsize=13, fontweight='medium', labelpad=8, color='#222222')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5, colors='#333333')
    ax.grid(False)
    
    ax.plot(1, 0, ">", color='#333333', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=7)
    ax.plot(0, 1, "^", color='#333333', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=7)
    
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    path2 = output_dir / f"doc{doc_id}_minimal_transparent.png"
    plt.savefig(path2, dpi=dpi, bbox_inches='tight', transparent=True, edgecolor='none')
    plt.close()
    print(f"   ✅ Saved: {path2}")
    
    # ========================================================================
    # VARIATION 3: THICKER LINE (more visible in small prints)
    # ========================================================================
    
    print("📈 Variation 3: Thicker Line (Small Print Ready)...")
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot(frames, scores_smooth, color='#1e3a8a', linewidth=2.8, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=14, fontweight='semibold', labelpad=10)
    ax.set_ylabel('Relevance Score', fontsize=14, fontweight='semibold', labelpad=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_color('#222222')
    
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.grid(False)
    
    ax.plot(1, 0, ">", color='#222222', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=8)
    ax.plot(0, 1, "^", color='#222222', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=8)
    
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    path3 = output_dir / f"doc{doc_id}_minimal_thick.png"
    plt.savefig(path3, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Saved: {path3}")
    
    # ========================================================================
    # VARIATION 4: BLACK LINE (Classic B&W for journals)
    # ========================================================================
    
    print("📈 Variation 4: Black Line (B&W Journal Ready)...")
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot(frames, scores_smooth, color='#000000', linewidth=1.8, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=13, fontweight='medium', labelpad=8)
    ax.set_ylabel('Relevance Score', fontsize=13, fontweight='medium', labelpad=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    ax.grid(False)
    
    ax.plot(1, 0, ">", color='#000000', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=7)
    ax.plot(0, 1, "^", color='#000000', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=7)
    
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    path4 = output_dir / f"doc{doc_id}_minimal_black.png"
    plt.savefig(path4, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Saved: {path4}")
    
    # ========================================================================
    # VARIATION 5: BLACK LINE TRANSPARENT (B&W + Transparent)
    # ========================================================================
    
    print("📈 Variation 5: Black Line Transparent...")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    
    ax.plot(frames, scores_smooth, color='#000000', linewidth=1.8, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=13, fontweight='medium', labelpad=8, color='#000000')
    ax.set_ylabel('Relevance Score', fontsize=13, fontweight='medium', labelpad=8, color='#000000')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5, colors='#000000')
    ax.grid(False)
    
    ax.plot(1, 0, ">", color='#000000', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=7)
    ax.plot(0, 1, "^", color='#000000', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=7)
    
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    path5 = output_dir / f"doc{doc_id}_minimal_black_transparent.png"
    plt.savefig(path5, dpi=dpi, bbox_inches='tight', transparent=True, edgecolor='none')
    plt.close()
    print(f"   ✅ Saved: {path5}")
    
    # ========================================================================
    # VARIATION 6: WIDE FORMAT (for two-column papers)
    # ========================================================================
    
    print("📈 Variation 6: Wide Format (Two-Column Paper)...")
    
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot(frames, scores_smooth, color='#1a56db', linewidth=1.8, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=11, fontweight='medium', labelpad=6)
    ax.set_ylabel('Relevance Score', fontsize=11, fontweight='medium', labelpad=6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=4)
    ax.grid(False)
    
    ax.plot(1, 0, ">", color='#333333', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=6)
    ax.plot(0, 1, "^", color='#333333', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=6)
    
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    path6 = output_dir / f"doc{doc_id}_minimal_wide.png"
    plt.savefig(path6, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Saved: {path6}")
    
    # ========================================================================
    # VARIATION 7: PDF/SVG VECTOR FORMAT
    # ========================================================================
    
    print("📈 Variation 7: Vector Formats (PDF & SVG)...")
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot(frames, scores_smooth, color='#1a56db', linewidth=2.0, solid_capstyle='round')
    
    ax.set_xlabel('Frame Index', fontsize=13, fontweight='medium', labelpad=8)
    ax.set_ylabel('Relevance Score', fontsize=13, fontweight='medium', labelpad=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    ax.grid(False)
    
    ax.plot(1, 0, ">", color='#333333', transform=ax.get_yaxis_transform(), 
            clip_on=False, markersize=7)
    ax.plot(0, 1, "^", color='#333333', transform=ax.get_xaxis_transform(), 
            clip_on=False, markersize=7)
    
    ax.set_xlim(frames[0] - x_pad, frames[-1] + x_pad)
    ax.set_ylim(scores_smooth.min() - y_pad, scores_smooth.max() + y_pad)
    
    plt.tight_layout()
    
    # Save as PDF (vector - infinitely scalable)
    path7_pdf = output_dir / f"doc{doc_id}_minimal_vector.pdf"
    plt.savefig(path7_pdf, bbox_inches='tight', facecolor='white', edgecolor='none', format='pdf')
    print(f"   ✅ Saved: {path7_pdf}")
    
    # Save as SVG (vector - editable in Illustrator/Inkscape)
    path7_svg = output_dir / f"doc{doc_id}_minimal_vector.svg"
    plt.savefig(path7_svg, bbox_inches='tight', facecolor='white', edgecolor='none', format='svg')
    print(f"   ✅ Saved: {path7_svg}")
    
    plt.close()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("✅ ALL VARIATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"📐 PNG Resolution: {dpi} DPI")
    print("\n📊 Generated files:")
    print(f"   1. {path1.name:<40} - White background (standard)")
    print(f"   2. {path2.name:<40} - Transparent PNG (customizable)")
    print(f"   3. {path3.name:<40} - Thicker line (small prints)")
    print(f"   4. {path4.name:<40} - Black line (B&W journals)")
    print(f"   5. {path5.name:<40} - Black + Transparent")
    print(f"   6. {path6.name:<40} - Wide format (two-column)")
    print(f"   7. {path7_pdf.name:<40} - PDF Vector (scalable)")
    print(f"   8. {path7_svg.name:<40} - SVG Vector (editable)")
    print("\n💡 RECOMMENDATIONS:")
    print("   • For LaTeX/Word papers: Use PDF or SVG (vector formats)")
    print("   • For PowerPoint/Slides: Use PNG with white background")
    print("   • For custom backgrounds: Use transparent PNG")
    print("   • For B&W printing: Use black line versions")
    print("=" * 70)
    
    return {
        'white': str(path1),
        'transparent': str(path2),
        'thick': str(path3),
        'black': str(path4),
        'black_transparent': str(path5),
        'wide': str(path6),
        'pdf': str(path7_pdf),
        'svg': str(path7_svg),
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    METADATA_PATH = "/home/train01/aks/Frame_selection/single_algo_tester/vmme/metadata_vmme.json"
    SCORES_PATH = "/home/train01/aks/Frame_selection/outscores/videomme/blip/scores.json"
    FRAMES_PATH = "/home/train01/aks/Frame_selection/outscores/videomme/blip/frames.json"
    
    # ========================================================================
    # SPECIFY YOUR DOCUMENT ID
    # ========================================================================
    
    DOC_ID = 2149  # <-- CHANGE THIS
    
    # ========================================================================
    # GENERATE ALL VARIATIONS
    # ========================================================================
    
    paths = generate_minimal_publication_plot(
        doc_id=DOC_ID,
        metadata_path=METADATA_PATH,
        scores_path=SCORES_PATH,
        frames_path=FRAMES_PATH,
        output_dir=f"publication_plots_doc{DOC_ID}",
        dpi=600  # High resolution
    )
    
    print("\n🎉 Done! Check your output folder for all variations.")