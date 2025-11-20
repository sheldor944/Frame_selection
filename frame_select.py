# import heapq
# import json
# import numpy as np
# import argparse
# import os

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Extract Video Feature')

#     parser.add_argument('--dataset_name', type=str, default='videomme', help='support longvideobench and videomme')
#     parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
#     parser.add_argument('--score_path', type=str, default='./outscores/videomme/blip/scores.json')
#     parser.add_argument('--frame_path', type=str, default='./outscores/videomme/blip/frames.json')
#     parser.add_argument('--max_num_frames', type=int, default=16)
#     parser.add_argument('--ratio', type=int, default=1)
#     parser.add_argument('--t1', type=int, default=0.8)
#     parser.add_argument('--t2', type=int, default=-100)
#     parser.add_argument('--all_depth', type=int, default=5)
#     parser.add_argument('--output_file', type=str, default='./selected_frames')
#     parser.add_argument('--num_videos', type=int, default=None, help='Number of videos to process (default: all)')

#     return parser.parse_args()

# def meanstd(len_scores, dic_scores, n, fns,t1,t2,all_depth):
#         split_scores = []
#         split_fn = []
#         no_split_scores = []
#         no_split_fn = []
#         i= 0
#         for dic_score, fn in zip(dic_scores, fns):
#                 # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
#                 score = dic_score['score']
#                 depth = dic_score['depth']
#                 mean = np.mean(score)
#                 std = np.std(score)

#                 top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
#                 top_score = [score[t] for t in top_n]
#                 # print(f"split {i}: ",len(score))
#                 i += 1
#                 mean_diff = np.mean(top_score) - mean
#                 if mean_diff > t1 and std > t2:
#                         no_split_scores.append(dic_score)
#                         no_split_fn.append(fn)
#                 elif depth < all_depth:
#                 # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
#                         score1 = score[:len(score)//2]
#                         score2 = score[len(score)//2:]
#                         fn1 = fn[:len(score)//2]
#                         fn2 = fn[len(score)//2:]                       
#                         split_scores.append(dict(score=score1,depth=depth+1))
#                         split_scores.append(dict(score=score2,depth=depth+1))
#                         split_fn.append(fn1)
#                         split_fn.append(fn2)
#                 else:
#                         no_split_scores.append(dic_score)
#                         no_split_fn.append(fn)
#         if len(split_scores) > 0:
#                 all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
#         else:
#                 all_split_score = []
#                 all_split_fn = []
#         all_split_score = no_split_scores + all_split_score
#         all_split_fn = no_split_fn + all_split_fn


#         return all_split_score, all_split_fn

# # def main(args):
# #     max_num_frames = args.max_num_frames
# #     ratio = args.ratio
# #     t1 = args.t1
# #     t2 = args.t2
# #     all_depth = args.all_depth
# #     outs = []
# #     segs = []

# #     with open(args.score_path) as f:
# #         itm_outs = json.load(f)
# #     with open(args.frame_path) as f:
# #         fn_outs = json.load(f)

# #     if args.num_videos is not None:
# #         num_videos_to_process = min(args.num_videos, len(itm_outs))
# #         itm_outs = itm_outs[:num_videos_to_process]
# #         fn_outs = fn_outs[:num_videos_to_process]
# #         print(f"🎯 DEMO MODE: Processing first {num_videos_to_process} videos only")
    
# #     print(f"Total videos to process: {len(itm_outs)}\n")


# #     if not os.path.exists(os.path.join(args.output_file,args.dataset_name)):
# #         os.mkdir(os.path.join(args.output_file,args.dataset_name))
# #     out_score_path = os.path.join(args.output_file,args.dataset_name,args.extract_feature_model)
# #     if not os.path.exists(out_score_path):
# #         os.mkdir(out_score_path)

# #     for itm_out,fn_out in zip(itm_outs,fn_outs):
# #         nums = int(len(itm_out)/ratio)
# #         new_score = [itm_out[num*ratio] for num in range(nums)]
# #         new_fnum = [fn_out[num*ratio] for num in range(nums)]
# #         score = new_score
# #         fn = new_fnum
# #         num = max_num_frames
# #         if len(score) >= num:
# #             normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
# #             a, b = meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
# #             segs.append(len(a))
# #             out = []
# #             if len(score) >= num:
# #                 for s,f in zip(a,b): 
# #                     f_num = int(num / 2**(s['depth']))
# #                     topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
# #                     f_nums = [f[t] for t in topk]
# #                     out.extend(f_nums)
# #             out.sort()
# #             outs.append(out)
# #         else:
# #             outs.append(fn)

# #     score_path = os.path.join(out_score_path,'selected_frames_demo.json')
# #     with open(score_path,'w') as f:
# #         json.dump(outs,f)



# def main(args):
#     max_num_frames = args.max_num_frames
#     ratio = args.ratio
#     t1 = args.t1
#     t2 = args.t2
#     all_depth = args.all_depth
#     outs = []
#     segs = []

#     print("=" * 60)
#     print("Recursive Frame Sampling")
#     print("=" * 60)
#     print(f"Dataset: {args.dataset_name}")
#     print(f"Feature Model: {args.extract_feature_model}")
#     print(f"Max Frames: {max_num_frames}")
#     print(f"Ratio: {ratio}")
#     print(f"Threshold t1: {t1}")
#     print(f"Threshold t2: {t2}")
#     print(f"Max Depth: {all_depth}")
#     print("=" * 60)

#     print(f"\nLoading scores from: {args.score_path}")
#     with open(args.score_path) as f:
#         itm_outs = json.load(f)
    
#     print(f"Loading frames from: {args.frame_path}")
#     with open(args.frame_path) as f:
#         fn_outs = json.load(f)

#     # Handle demo mode
#     if args.num_videos is not None:
#         num_videos_to_process = min(args.num_videos, len(itm_outs))
#         itm_outs = itm_outs[:num_videos_to_process]
#         fn_outs = fn_outs[:num_videos_to_process]
#         print(f"\n🎯 DEMO MODE: Processing first {num_videos_to_process} videos only")
    
#     print(f"Total videos loaded: {len(itm_outs)}")
#     print(f"Videos to process: {len(itm_outs)}\n")

#     if not os.path.exists(os.path.join(args.output_file, args.dataset_name)):
#         os.mkdir(os.path.join(args.output_file, args.dataset_name))
#     out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
#     if not os.path.exists(out_score_path):
#         os.mkdir(out_score_path)

#     # Statistics tracking
#     total_input_frames = 0
#     total_output_frames = 0
#     video_stats = []

#     for idx, (itm_out, fn_out) in enumerate(zip(itm_outs, fn_outs)):
#         if len(itm_outs) <= 20:
#             print(f"Processing video {idx + 1}/{len(itm_outs)}...")
#         elif (idx + 1) % 100 == 0:
#             print(f"Processing video {idx + 1}/{len(itm_outs)}...")
        
#         nums = int(len(itm_out) / ratio)
#         new_score = [itm_out[num * ratio] for num in range(nums)]
#         new_fnum = [fn_out[num * ratio] for num in range(nums)]
#         score = new_score
#         fn = new_fnum
#         num = max_num_frames
        
#         # Track input frames
#         input_frame_count = len(score)
#         total_input_frames += input_frame_count
        
#         if len(score) >= num:
#             normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
#             a, b = meanstd(len(score), [dict(score=normalized_data, depth=0)], num, [fn], t1, t2, all_depth)
#             segs.append(len(a))
#             out = []
#             if len(score) >= num:
#                 for s, f in zip(a, b): 
#                     f_num = int(num / 2**(s['depth']))
#                     topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
#                     f_nums = [f[t] for t in topk]
#                     out.extend(f_nums)
#             out.sort()
#             outs.append(out)
#             output_frame_count = len(out)
#         else:
#             outs.append(fn)
#             output_frame_count = len(fn)
        
#         # Track output frames and video stats
#         total_output_frames += output_frame_count
#         video_stats.append({
#             'video_id': idx,
#             'input_frames': input_frame_count,
#             'output_frames': output_frame_count,
#             'compression_ratio': input_frame_count / output_frame_count if output_frame_count > 0 else 0,
#             'num_segments': segs[-1] if segs and len(score) >= num else 1
#         })

#     score_path = os.path.join(out_score_path, 'selected_frames_recursive_16_frames.json')
#     with open(score_path, 'w') as f:
#         json.dump(outs, f)
    
#     # Print comprehensive statistics
#     print(f"\n{'=' * 60}")
#     print(f"✅ Processing complete!")
#     print(f"Selected frames saved to: {score_path}")
#     print(f"\n{'=' * 60}")
#     print("STATISTICS")
#     print("=" * 60)
    
#     # Basic stats
#     frame_counts = [len(frames) for frames in outs]
#     print(f"\n📊 Frame Selection:")
#     print(f"  Videos processed: {len(outs)}")
#     print(f"  Total input frames: {total_input_frames:,}")
#     print(f"  Total output frames: {total_output_frames:,}")
#     print(f"  Overall compression ratio: {total_input_frames / total_output_frames:.2f}x")
#     print(f"  Avg frames selected per video: {np.mean(frame_counts):.2f}")
#     print(f"  Min frames selected: {np.min(frame_counts)}")
#     print(f"  Max frames selected: {np.max(frame_counts)}")
#     print(f"  Median frames selected: {np.median(frame_counts):.2f}")
#     print(f"  Std dev: {np.std(frame_counts):.2f}")
    
#     # Segment statistics (if available)
#     if segs:
#         print(f"\n🔀 Segmentation:")
#         print(f"  Avg segments per video: {np.mean(segs):.2f}")
#         print(f"  Min segments: {np.min(segs)}")
#         print(f"  Max segments: {np.max(segs)}")
#         print(f"  Median segments: {np.median(segs):.2f}")
    
#     # Compression ratio distribution
#     compression_ratios = [stat['compression_ratio'] for stat in video_stats if stat['compression_ratio'] > 0]
#     if compression_ratios:
#         print(f"\n📉 Compression Ratios:")
#         print(f"  Mean: {np.mean(compression_ratios):.2f}x")
#         print(f"  Min: {np.min(compression_ratios):.2f}x")
#         print(f"  Max: {np.max(compression_ratios):.2f}x")
#         print(f"  Median: {np.median(compression_ratios):.2f}x")
    
#     # Frame distribution
#     print(f"\n📈 Frame Distribution:")
#     bins = [0, 32, 64, 96, 128, float('inf')]
#     labels = ['0-32', '33-64', '65-96', '97-128', '128+']
#     hist, _ = np.histogram(frame_counts, bins=bins)
#     for label, count in zip(labels, hist):
#         percentage = (count / len(frame_counts)) * 100
#         print(f"  {label} frames: {count} videos ({percentage:.1f}%)")
    
#     # Detailed video stats (for demo mode, show first few)
#     if args.num_videos is not None and len(video_stats) <= 20:
#         print(f"\n📋 Per-Video Details:")
#         print(f"{'Video':<8} {'Input':<8} {'Output':<8} {'Ratio':<8} {'Segments':<10}")
#         print("-" * 50)
#         for stat in video_stats:
#             print(f"{stat['video_id']:<8} {stat['input_frames']:<8} {stat['output_frames']:<8} "
#                   f"{stat['compression_ratio']:<8.2f} {stat['num_segments']:<10}")
    
#     print("=" * 60)
    
#     # Optionally save detailed stats to JSON
#     stats_path = os.path.join(out_score_path, 'statistics_recursive.json')
#     stats_summary = {
#         'total_videos': len(outs),
#         'total_input_frames': total_input_frames,
#         'total_output_frames': total_output_frames,
#         'overall_compression_ratio': total_input_frames / total_output_frames if total_output_frames > 0 else 0,
#         'avg_frames_selected': float(np.mean(frame_counts)),
#         'min_frames': int(np.min(frame_counts)),
#         'max_frames': int(np.max(frame_counts)),
#         'median_frames': float(np.median(frame_counts)),
#         'std_frames': float(np.std(frame_counts)),
#         'avg_segments': float(np.mean(segs)) if segs else 0,
#         'frame_distribution': {
#             label: int(count) for label, count in zip(labels, hist)
#         },
#         'per_video_stats': video_stats
#     }
    
#     with open(stats_path, 'w') as f:
#         json.dump(stats_summary, f, indent=2)
    
#     print(f"📊 Detailed statistics saved to: {stats_path}\n")





# if __name__ == '__main__':
#     args = parse_arguments()
#     main(args)


# import heapq
# import json
# import numpy as np
# import argparse
# import os

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Extract Video Feature')

#     parser.add_argument('--dataset_name', type=str, default='videomme', help='support longvideobench and videomme')
#     parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
#     parser.add_argument('--score_path', type=str, default='./outscores/videomme/blip/scores.json')
#     parser.add_argument('--frame_path', type=str, default='./outscores/videomme/blip/frames.json')
#     parser.add_argument('--max_num_frames', type=int, default=20)
#     parser.add_argument('--ratio', type=int, default=1)
#     parser.add_argument('--t1', type=int, default=0.8)
#     parser.add_argument('--t2', type=int, default=-100)
#     parser.add_argument('--all_depth', type=int, default=5)
#     parser.add_argument('--output_file', type=str, default='./selected_frames')

#     return parser.parse_args()

# def meanstd(len_scores, dic_scores, n, fns,t1,t2,all_depth):
#         split_scores = []
#         split_fn = []
#         no_split_scores = []
#         no_split_fn = []
#         i= 0
#         for dic_score, fn in zip(dic_scores, fns):
#                 # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
#                 score = dic_score['score']
#                 depth = dic_score['depth']
#                 mean = np.mean(score)
#                 std = np.std(score)

#                 top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
#                 top_score = [score[t] for t in top_n]
#                 # print(f"split {i}: ",len(score))
#                 i += 1
#                 mean_diff = np.mean(top_score) - mean
#                 if mean_diff > t1 and std > t2:
#                         no_split_scores.append(dic_score)
#                         no_split_fn.append(fn)
#                 elif depth < all_depth:
#                 # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
#                         score1 = score[:len(score)//2]
#                         score2 = score[len(score)//2:]
#                         fn1 = fn[:len(score)//2]
#                         fn2 = fn[len(score)//2:]                       
#                         split_scores.append(dict(score=score1,depth=depth+1))
#                         split_scores.append(dict(score=score2,depth=depth+1))
#                         split_fn.append(fn1)
#                         split_fn.append(fn2)
#                 else:
#                         no_split_scores.append(dic_score)
#                         no_split_fn.append(fn)
#         if len(split_scores) > 0:
#                 all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
#         else:
#                 all_split_score = []
#                 all_split_fn = []
#         all_split_score = no_split_scores + all_split_score
#         all_split_fn = no_split_fn + all_split_fn


#         return all_split_score, all_split_fn

# def main(args):
#     max_num_frames = args.max_num_frames
#     ratio = args.ratio
#     t1 = args.t1
#     t2 = args.t2
#     all_depth = args.all_depth
#     outs = []
#     segs = []

#     with open(args.score_path) as f:
#         itm_outs = json.load(f)
#     with open(args.frame_path) as f:
#         fn_outs = json.load(f)

#     if not os.path.exists(os.path.join(args.output_file,args.dataset_name)):
#         os.makedirs(os.path.join(args.output_file,args.dataset_name))
#     out_score_path = os.path.join(args.output_file,args.dataset_name,args.extract_feature_model)
#     if not os.path.exists(out_score_path):
#         os.makedirs(out_score_path)

#     for itm_out,fn_out in zip(itm_outs,fn_outs):
#         nums = int(len(itm_out)/ratio)
#         new_score = [itm_out[num*ratio] for num in range(nums)]
#         new_fnum = [fn_out[num*ratio] for num in range(nums)]
#         score = new_score
#         fn = new_fnum
#         num = max_num_frames
        
#         if len(score) >= num:
#             normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
#             a, b = meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
#             segs.append(len(a))
#             out = []
            
#             total_frames_needed = num
#             total_segments = len(a)
            
#             for s, f in zip(a, b): 
#                 # Calculate frames for this segment proportionally
#                 f_num = max(1, int(num / 2**(s['depth'])))  # Ensure at least 1 frame
                
#                 # Don't try to select more frames than available in segment
#                 f_num = min(f_num, len(s['score']))
                
#                 if f_num > 0 and len(s['score']) > 0:
#                     topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
#                     f_nums = [f[t] for t in topk]
#                     out.extend(f_nums)
            
#             # If we don't have enough frames, sample uniformly
#             if len(out) < num and len(fn) >= num:
#                 indices = np.linspace(0, len(fn)-1, num, dtype=int)
#                 out = [fn[i] for i in indices]
            
#             out.sort()
#             # Trim to exact number if we have too many
#             outs.append(out[:num])
#         else:
#             # Uniformly sample if not enough frames
#             if len(fn) < num:
#                 outs.append(fn)
#             else:
#                 indices = np.linspace(0, len(fn)-1, num, dtype=int)
#                 outs.append([fn[i] for i in indices])

#     score_path = os.path.join(out_score_path,'selected_frames_ASK_20_updated.json')
#     with open(score_path,'w') as f:
#         json.dump(outs,f)


# if __name__ == '__main__':
#     args = parse_arguments()
#     main(args)


import heapq
import json
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='videomme', help='support longvideobench and videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip')
    parser.add_argument('--score_path', type=str, default='./outscores/videomme/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/videomme/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='./selected_frames')

    return parser.parse_args()


# -------------------------------
# RECURSIVE SPLITTING (UNCHANGED)
# -------------------------------
def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):

    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []

    for dic_score, fn in zip(dic_scores, fns):

        score = dic_score['score']
        depth = dic_score['depth']

        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]

        mean_diff = np.mean(top_score) - mean

        # Stop condition (same as AKS)
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

        # Recursive split
        elif depth < all_depth:
            mid = len(score) // 2

            score1 = score[:mid]
            score2 = score[mid:]

            fn1 = fn[:mid]
            fn2 = fn[mid:]

            split_scores.append(dict(score=score1, depth=depth+1))
            split_scores.append(dict(score=score2, depth=depth+1))

            split_fn.append(fn1)
            split_fn.append(fn2)

        else:
            # max depth reached
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

    if len(split_scores) > 0:
        rec_scores, rec_fn = meanstd(
            len_scores,
            split_scores,
            n,
            split_fn,
            t1, t2, all_depth
        )
    else:
        rec_scores, rec_fn = [], []

    final_scores = no_split_scores + rec_scores
    final_fn = no_split_fn + rec_fn

    return final_scores, final_fn



# -------------------------------
# AKS MAIN LOGIC WITH DEPTH FIX
# -------------------------------
def main(args):
    max_num_frames = args.max_num_frames
    ratio = args.ratio
    t1 = args.t1
    t2 = args.t2
    all_depth = args.all_depth

    outs = []

    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    if not os.path.exists(os.path.join(args.output_file,args.dataset_name)):
        os.makedirs(os.path.join(args.output_file,args.dataset_name))

    out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.makedirs(out_score_path)

    for itm_out, fn_out in zip(itm_outs, fn_outs):

        nums = int(len(itm_out) / ratio)
        score = [itm_out[i*ratio] for i in range(nums)]
        fn = [fn_out[i*ratio] for i in range(nums)]

        num = max_num_frames

        if len(score) >= num:

            normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))

            a, b = meanstd(
                len(score),
                [dict(score=normalized_data, depth=0)],
                num,
                [fn],
                t1, t2, all_depth
            )

            out = []

            # -------- EXACT AKS ALLOCATION (NO FORCE PICK) --------
            for s, f in zip(a, b):

                depth = s['depth']
                f_num = int(num / (2 ** depth))

                # depth reduction if f_num == 0
                while f_num == 0 and depth > 0:
                    depth -= 1
                    f_num = int(num / (2 ** depth))

                if f_num > 0 and len(s['score']) > 0:
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    out.extend([f[i] for i in topk])

            out.sort()
            outs.append(out)

        else:
            outs.append(fn)

    score_path = os.path.join(out_score_path, 'selected_frames_AKS_8_exact.json')
    with open(score_path, 'w') as f:
        json.dump(outs, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
