"""
多模态TStar框架脚本,用于同时利用视觉和文本信息进行视频检索和问答
"""

import os
import argparse
import json
import re
import copy
import torch
import numpy as np
import sys
import datetime
from TStar.TStarMultiModalFramework import TStarMultiModalFramework
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloWorldInterface
import time
from exp_logging import ExperimentRecorder

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VSI_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', '..'))
LVB_ROOT = '/nfs-stor/jianxiang.he/continual-vsi/data/LongVideoBench'
YOLO_WORLD_ROOT = os.path.join(VSI_ROOT, 'third_party', 'YOLO-World')
YOLO_WEIGHTS = '/l/users/jianxiang.he/YOLO-World/weights'

class DebugLogger:
    """调试模式下的日志记录器，同时输出到控制台和文件"""
    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        self.log_file = None
        
        if self.log_file_path:
            try:
                self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
                # 写入文件头
                header = f"""
=== TStar多模态视频检索调试日志 ===
开始时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"="*60}

"""
                self.log_file.write(header)
                self.log_file.flush()
            except Exception as e:
                print(f"无法创建日志文件 {self.log_file_path}: {e}")
                self.log_file = None
    
    def write(self, text):
        # 始终输出到控制台
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # 如果有日志文件，也写入文件
        if self.log_file:
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except Exception as e:
                self.original_stdout.write(f"[日志写入错误: {e}]\n")
    
    def flush(self):
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            # 写入结束信息
            footer = f"""

{"="*60}
结束时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
=== 调试日志结束 ===
"""
            self.log_file.write(footer)
            self.log_file.close()
            self.log_file = None
        
        # 恢复原始stdout
        sys.stdout = self.original_stdout

def setup_debug_logging(args):
    """设置调试模式的日志记录"""
    if not args.debug_mode:
        return None
    
    # 创建日志目录
    log_dir = os.path.join(args.results_dir, 'debug_log')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = args.device.replace(':', '-')
    log_filename = f"debug_{timestamp}_w{args.text_weight}_device{device_name}_dataset{args.dataset}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    # 创建调试日志记录器
    debug_logger = DebugLogger(log_file_path)
    
    # 重定向stdout到我们的日志记录器
    sys.stdout = debug_logger
    
    # 输出配置信息
    print(f"=== 调试模式启动 ===")
    print(f"日志文件: {log_file_path}")
    print(f"参数配置:")
    print(f"  - 文本权重: {args.text_weight}")
    print(f"  - 设备: {args.device}")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 搜索帧数: {args.search_nframes}")
    print(f"  - 置信度阈值: {args.confidence_threshold}")
    print(f"  - 搜索预算: {args.search_budget}")
    print(f"  - 数据路径: {args.data_path}")
    print(f"{'='*50}\n")
    
    return debug_logger

def load_lvbench_subtitles(subtitle_path):
    """加载LongVideoBench字幕"""
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_videomme_srt(srt_path):
    """解析VideoMME的SRT字幕文件"""
    subtitles = []
    if not os.path.exists(srt_path):
        return subtitles
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if time_match:
                start = time_match.group(1)
                end = time_match.group(2)
                def to_sec(t):
                    h, m, s_ms = t.split(':')
                    s, ms = s_ms.split(',')
                    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
                start_sec = to_sec(start)
                end_sec = to_sec(end)
                text = ' '.join([re.sub(r'<.*?>', '', l) for l in lines[2:]]).strip()
                subtitles.append({'start': start_sec, 'end': end_sec, 'text': text})
    return subtitles


def load_videomme_subtitles(video_id, subtitle_root):
    """加载VideoMME字幕"""
    srt_path = os.path.join(subtitle_root, f"{video_id}.srt")
    return parse_videomme_srt(srt_path)


def process_TStar_onVideo_multimodal(args, data_item,
                                     yolo_scorer: YoloWorldInterface,
                                     grounder: TStarUniversalGrounder) -> dict:
    """使用多模态信息处理单个视频的搜索和问答任务
    
    参数:
        args: 包含各种配置参数的对象
        data_item: 包含视频信息的数据项
        yolo_scorer: YOLO目标检测器接口
        grounder: 通用目标定位器接口
        
    返回:
        dict: 包含处理结果的字典,包括:
            - video_path: 视频路径
            - grounding_objects: 定位到的目标对象
            - keyframe_timestamps: 关键帧时间戳
            - frame_distribution: 帧分布概率
    """
    import time
    total_start_time = time.time()
    # 获取字幕文件路径
    if args.dataset == 'VideoMME':
        subtitle_path = os.path.join(args.subtitle_root, f"{data_item['video_id']}.srt")
    else:
        subtitle_path = os.path.join(args.subtitle_root, data_item['video_id'] + "_en.json")
    # 原实验中缺字幕的视频会直接报错跳过；打开此开关则退化为纯视觉搜索
    if args.visual_only_if_no_subtitle and not os.path.exists(subtitle_path):
        print(f"字幕文件不存在，仅使用视觉分支: {subtitle_path}")
        subtitle_path = ""

    # 调试模式下输出更多信息
    if args.debug_mode:
        print(f"\n===== 处理视频详情 =====")
        print(f"视频ID: {data_item['video_id']}")
        print(f"视频路径: {data_item['video_path']}")
        print(f"字幕路径: {subtitle_path}")
        print(f"问题: {data_item['question']}")
        print(f"问题类别: {data_item.get('question_category', 'N/A')}")
        
        # 如果存在position，输出位置信息
        if 'position' in data_item and data_item['position']:
            print(f"Ground truth position: {data_item['position']}")
            
            # 尝试获取视频FPS
            try:
                from decord import VideoReader, cpu
                video_reader = VideoReader(data_item['video_path'], ctx=cpu(0))
                fps = video_reader.get_avg_fps()
                print(f"视频FPS: {fps}")
                
                #     # 检查position的类型和值范围来判断它是时间戳还是帧号
                #     if isinstance(data_item['position'], list):
                #         # 检查列表中的值是否可能是帧号
                #         position_is_frame = all(pos > 30 for pos in data_item['position'] if isinstance(pos, (int, float)))
                        
                #         if position_is_frame:
                #             # 如果position很可能已经是帧号，直接使用
                #             frame_positions = data_item['position']
                #             print(f"Ground truth位置已经是帧号: {frame_positions}")
                #         else:
                #             # 如果position是时间戳（秒），转换为帧号
                #             frame_positions = [int(pos * fps) if isinstance(pos, (int, float)) else None 
                #                               for pos in data_item['position']]
                #             print(f"时间戳转换的帧位置: {frame_positions}")
                #     elif isinstance(data_item['position'], (int, float)):
                #         # 对单个值进行同样的判断
                #         if data_item['position'] > 30:  # 假设大于30的值很可能是帧号
                #             frame_position = data_item['position']
                #             print(f"Ground truth位置已经是帧号: {frame_position}")
                #         else:
                #             frame_position = int(data_item['position'] * fps)
                #             print(f"时间戳转换的帧位置: {frame_position}")
            except Exception as e:
                print(f"获取视频FPS失败: {e}")

    # 初始化多模态TStar框架
    TStar_framework = TStarMultiModalFramework(
        grounder=grounder,  # 目标定位器
        yolo_scorer=yolo_scorer,  # YOLO检测器
        video_path=data_item['video_path'],  # 视频路径
        subtitle_path=subtitle_path,  # 传递路径字符串
        question=data_item['question'],  # 问题
        options=data_item['options'],  # 选项
        text_weight=args.text_weight,  # 文本相似度权重
        search_nframes=args.search_nframes,  # 返回的top帧数
        grid_rows=args.grid_rows,  # 图像网格的行数
        grid_cols=args.grid_cols,  # 图像网格的列数
        output_dir=args.output_dir,  # 输出目录
        confidence_threshold=args.confidence_threshold,  # YOLO检测置信度阈值
        search_budget=args.search_budget,  # 搜索帧的最大比例
        prefix=args.prefix,  # 输出文件名前缀
        device=args.device  # 设备
    )

    # 运行框架主流程 - 使用完整的多模态流程
    # target_objects, cue_objects = TStar_framework.get_grounded_objects(args.prompt_type, args.upload_video)  # 获取定位目标
    target_objects = data_item['grounding_objects']['target_objects']
    cue_objects = data_item['grounding_objects']['cue_objects']
    # relations = data_item['grounding_objects'].get('relations', [])

    # Initialize Searching Targets to VSLS Seacher
    # video_searcher = TStar_framework.set_searching_targets(target_objects, cue_objects, relations)
    video_searcher = TStar_framework.set_searching_targets(target_objects, cue_objects)
    # video_searcher.relation_alpha = args.relation_alpha 

    # 计算文本相似度分数
    text_scores = TStar_framework.compute_text_similarity_score(TStar_framework.question)
    
    # 融合视觉和文本分数 - 替换update_frame_distribution方法
    original_update_frame_distribution = video_searcher.update_frame_distribution
    
    def new_update_frame_distribution(sampled_frame_indices, confidence_maps, detected_objects_maps, bbox_maps):
        # 首先调用原始的更新方法获取视觉分数
        frame_confidences, frame_detected_objects = original_update_frame_distribution(
            sampled_frame_indices, confidence_maps, detected_objects_maps, bbox_maps
        )
        
        # 获取当前的视觉分数分布
        visual_scores = video_searcher.score_distribution.copy()
        
        # 重新采样文本分数以匹配视觉分数的长度
        # 视觉分数长度基于1fps采样：video_searcher.total_frame_num
        # 文本分数长度基于原始帧数：len(text_scores)
        visual_length = len(visual_scores)
        text_length = len(text_scores)
        
        # 将文本分数重采样到与视觉分数相同的长度
        if text_length != visual_length:
            # 使用线性插值重采样
            from scipy.interpolate import interp1d
            
            # 创建原始的索引
            original_indices = np.linspace(0, text_length - 1, text_length)
            # 创建目标索引  
            target_indices = np.linspace(0, text_length - 1, visual_length)
            
            # 使用插值函数重采样
            f = interp1d(original_indices, text_scores, kind='linear', fill_value=1, bounds_error=False)
            resampled_text_scores = f(target_indices)
        else:
            resampled_text_scores = text_scores
        
        # 对视觉分数和重采样后的文本分数分别进行归一化处理
        # 视觉分数归一化
        visual_max = np.max(visual_scores)
        visual_min = np.min(visual_scores)
        if visual_max > visual_min:  # 防止除以零
            normalized_visual_scores = (visual_scores - visual_min) / (visual_max - visual_min)
        else:
            normalized_visual_scores = np.ones_like(visual_scores) * 0.5  # 如果所有分数相等，设为0.5
        
        # 文本分数归一化
        text_max = np.max(resampled_text_scores)
        text_min = np.min(resampled_text_scores)
        if text_max > text_min:  # 防止除以零
            normalized_text_scores = (resampled_text_scores - text_min) / (text_max - text_min)
        else:
            normalized_text_scores = np.ones_like(resampled_text_scores) * 0.5  # 如果所有分数相等，设为0.5
        
        print(f"文本权重: {TStar_framework.text_weight:.4f}, 文本最大分数: {text_max:.4f}")
        print(f"归一化前 - 视觉分数范围: {visual_min:.4f}-{visual_max:.4f}, 文本分数范围: {text_min:.4f}-{text_max:.4f}")
        print(f"重采样: 文本分数从 {text_length} 帧重采样到 {visual_length} 帧")
        
        # 使用归一化后的分数进行静态权重融合
        fused_scores = (1 - TStar_framework.text_weight) * normalized_visual_scores + TStar_framework.text_weight * normalized_text_scores
        
        # 更新score_distribution为融合后的分数
        video_searcher.score_distribution = fused_scores
        
        # 重新计算概率分布P
        if video_searcher.update_method == "spline":
            video_searcher.P = video_searcher.spline_keyframe_distribution(
                video_searcher.non_visiting_frames,
                video_searcher.score_distribution,
                len(video_searcher.score_distribution)
            )
        elif video_searcher.update_method == "gaussian":
            video_searcher.P = video_searcher.gaussioan_score_distribution(
                video_searcher.non_visiting_frames,
                video_searcher.score_distribution,
                len(video_searcher.score_distribution)
            )
        
        # 存储更新后的分布
        video_searcher.store_score_distribution()
        
        return frame_confidences, frame_detected_objects
        
    # 替换方法为融合版本
    video_searcher.update_frame_distribution = new_update_frame_distribution

    search_start_time = time.time()
    all_frames, time_stamps = TStar_framework.perform_search(video_searcher)  # 执行搜索
    search_end_time = time.time()

    # 收集结果
    num_iterations = TStar_framework.results.get('num_iterations', 0)
    total_end_time = time.time()
    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": TStar_framework.results.get('Searching_Objects', []),
        "keyframe_timestamps": TStar_framework.results.get('timestamps', []),
        "frame_distribution": video_searcher.P_history[-1],
        "score_distribution": video_searcher.score_distribution.tolist(),  # 保存融合后的分数分布
        "num_iterations": num_iterations,
        "search_time": search_end_time - search_start_time,
        "total_time": total_end_time - total_start_time
    }

    # 可选：保留每轮迭代后的采样概率分布
    if args.retain_sampling_history:
        result["sampling_probability_history"] = video_searcher.P_history
    
    # 在调试模式下保存更多分数信息
    if args.debug_mode:
        result.update({
            "score_distribution_info": {
                "score_length": len(video_searcher.score_distribution),
                "score_min": float(np.min(video_searcher.score_distribution)),
                "score_max": float(np.max(video_searcher.score_distribution)),
                "score_mean": float(np.mean(video_searcher.score_distribution)),
                "text_weight": TStar_framework.text_weight
            }
        })
    
    # 在调试模式下检查关键帧是否匹配position
    if args.debug_mode and 'position' in data_item and data_item['position'] and 'keyframe_timestamps' in result:
        try:
            from decord import VideoReader, cpu
            video_reader = VideoReader(data_item['video_path'], ctx=cpu(0))
            fps = video_reader.get_avg_fps()
            
            print("\n===== 关键帧与Ground Truth Position匹配分析 =====")
            keyframe_timestamps = result['keyframe_timestamps']
            keyframe_frames = [int(ts * fps) for ts in keyframe_timestamps]
            
            # 获取ground truth position
            gt_positions = data_item['position'] if isinstance(data_item['position'], list) else [data_item['position']]
            
            # 检查gt_positions中的值是否可能已经是帧号
            position_is_frame = all(pos > 30 for pos in gt_positions if isinstance(pos, (int, float)))
            
            if position_is_frame:
                # 如果已经是帧号，直接使用
                gt_frames = [int(pos) if isinstance(pos, (int, float)) else None for pos in gt_positions]
                print(f"使用原始Ground Truth帧位置: {gt_frames}")
            else:
                # 如果是时间戳（秒），转换为帧号
                gt_frames = [int(pos * fps) if isinstance(pos, (int, float)) else None for pos in gt_positions]
                print(f"时间戳转换的Ground Truth帧位置: {gt_frames}")
            
            print(f"关键帧时间戳: {keyframe_timestamps}")
            print(f"关键帧帧位置: {keyframe_frames}")
            
            # 计算每个关键帧与ground truth的距离
            threshold = 200  # 设置匹配阈值为200帧
            match_results = []
            
            for i, kf_frame in enumerate(keyframe_frames):
                best_match = {'gt_frame': None, 'distance': float('inf'), 'match': False}
                
                for gt_frame in gt_frames:
                    if gt_frame is not None:
                        distance = abs(kf_frame - gt_frame)
                        if distance < best_match['distance']:
                            best_match = {
                                'gt_frame': gt_frame,
                                'distance': distance,
                                'match': distance <= threshold
                            }
                
                match_results.append({
                    'keyframe_idx': i,
                    'keyframe_time': keyframe_timestamps[i],
                    'keyframe_frame': kf_frame,
                    'best_match_gt': best_match['gt_frame'],
                    'distance': best_match['distance'],
                    'match': best_match['match']
                })
                
                # 打印结果
                if best_match['gt_frame'] is not None:
                    match_status = "✓" if best_match['match'] else "✗"
                    print(f"关键帧 #{i+1} ({keyframe_timestamps[i]:.2f}s, 第{kf_frame}帧) -> " + 
                          f"最接近GT: 第{best_match['gt_frame']}帧, 距离: {best_match['distance']}帧 {match_status}")
            
            # 统计匹配情况
            matches = sum(1 for r in match_results if r['match'])
            print(f"\n匹配结果: {matches}/{len(keyframe_frames)} 关键帧与Ground Truth匹配 " + 
                  f"({matches/len(keyframe_frames)*100:.2f}%)")
            
            # 将匹配结果添加到返回数据中
            result['position_match_analysis'] = {
                'match_results': match_results,
                'matches': matches,
                'total': len(keyframe_frames),
                'match_rate': matches/len(keyframe_frames) if len(keyframe_frames) > 0 else 0,
                'threshold': threshold
            }
            
        except Exception as e:
            print(f"进行关键帧匹配分析时出错: {e}")

    return result


def analyze_results(results, args):
    """
    分析处理结果，比较关键帧与字幕匹配情况
    """
    subtitle_questions = [r for r in results if r.get("is_subtitle_question", False)]
    non_subtitle_questions = [r for r in results if not r.get("is_subtitle_question", False)]
    
    print("\n===== 结果分析 =====")
    print(f"总样本数: {len(results)}")
    print(f"字幕相关问题数: {len(subtitle_questions)}")
    print(f"非字幕相关问题数: {len(non_subtitle_questions)}")
    
    # 统计正确回答的数量
    correct_subtitle = sum(1 for r in subtitle_questions if r.get("correct", False))
    correct_non_subtitle = sum(1 for r in non_subtitle_questions if r.get("correct", False))
    
    if len(subtitle_questions) > 0:
        print(f"字幕相关问题正确率: {correct_subtitle/len(subtitle_questions)*100:.2f}%")
    if len(non_subtitle_questions) > 0:
        print(f"非字幕相关问题正确率: {correct_non_subtitle/len(non_subtitle_questions)*100:.2f}%")
    
    # 如果有position信息，可以进一步分析关键帧与实际位置的匹配情况
    print("\n分析关键帧与实际位置的匹配情况...")
    
    match_counts = {"exact": 0, "within_30_frames": 0, "no_match": 0}
    videos_analyzed = 0
    
    for item in subtitle_questions:
        if "position" in item and "keyframe_timestamps" in item:
            videos_analyzed += 1
            positions = item["position"]
            timestamps = item["keyframe_timestamps"]
            
            # 计算最佳匹配
            best_match_diff = float('inf')
            for pos in positions:
                for ts in timestamps:
                    # 假设position和timestamp都以秒为单位
                    diff = abs(pos - ts)
                    best_match_diff = min(best_match_diff, diff)
            
            # 根据差异分类
            if best_match_diff < 1.0:  # 1秒内视为精确匹配
                match_counts["exact"] += 1
            elif best_match_diff < 30/25:  # 假设25fps，30帧约等于1.2秒
                match_counts["within_30_frames"] += 1
            else:
                match_counts["no_match"] += 1
    
    if videos_analyzed > 0:
        print(f"分析了 {videos_analyzed} 个字幕相关视频样本")
        print(f"精确匹配率 (差异<1秒): {match_counts['exact']/videos_analyzed*100:.2f}%")
        print(f"大致匹配率 (差异<30帧): {match_counts['within_30_frames']/videos_analyzed*100:.2f}%")
        print(f"无匹配率: {match_counts['no_match']/videos_analyzed*100:.2f}%")


def main():
    """主函数:解析参数并执行批量视频处理"""
    parser = argparse.ArgumentParser()
    # 添加原始参数
    parser.add_argument('--text_weight', type=float, default=0.3,
                        help='文本相似度分数在融合中的权重')
    parser.add_argument('--dataset', type=str, default="LongVideoBench", help='用于处理的视频数据集')
    parser.add_argument('--data_path', type=str, default="obj_PTonly_LongVideoBench_gpt4_cot_vid1.json", help='数据路径')
    parser.add_argument('--grounding_dir', type=str, default=os.path.join(REPO_ROOT, 'data', 'object_grounding'), help='object grounding 结果所在目录（--data_path 相对于它）')
    parser.add_argument('--video_root', type=str, default=os.path.join(LVB_ROOT, 'videos'), help='视频根目录；grounding JSON 里的 video_path 会按文件名重定向到这里')
    parser.add_argument('--results_dir', type=str, default=os.path.join(REPO_ROOT, 'results'), help='保存批量处理结果的路径')
    parser.add_argument('--config_path', type=str, default=os.path.join(YOLO_WORLD_ROOT, 'configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'), help='YOLO配置文件路径')
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join(YOLO_WEIGHTS, 'yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth'), help='YOLO模型检查点路径')
    parser.add_argument('--device', type=str, default="cuda:0", help='模型推理的设备(例如"cuda:0"或"cpu")')
    parser.add_argument('--search_nframes', type=int, default=8, help='返回的top帧数')
    parser.add_argument('--grid_rows', type=int, default=4, help='图像网格的行数')
    parser.add_argument('--grid_cols', type=int, default=4, help='图像网格的列数')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO检测置信度阈值')
    parser.add_argument('--search_budget', type=float, default=1.0, help='搜索帧的最大比例')
    parser.add_argument('--output_dir', type=str, default='./output', help='保存输出的目录')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='输出文件名前缀')
    parser.add_argument('--backend', type=str, default='gpt4', help='用于定位的后端(gpt4或llava)')
    parser.add_argument('--prompt_type', type=str, default='cot', help='使用的提示类型')
    parser.add_argument('--save_batch', type=int, default=10, help='每N个条目保存批量结果到output_json')
    parser.add_argument('--num', type=int, default=0, help='要处理的视频数量')
    parser.add_argument('--upload_video', type=int, default=1, help='上传视频到OpenAI API')
    parser.add_argument('--update_method', type=str, default='spline', help='更新分布的方法')
    # parser.add_argument('--relation_alpha', type=float, default=0, help='关系分数的权重')
    parser.add_argument('--filter_subtitle', action='store_true',
                        help='是否过滤掉字幕相关的问题')
    parser.add_argument('--visual_only_if_no_subtitle', action='store_true',
                        help='字幕文件缺失时退化为纯视觉搜索（默认与原实验一致：报错跳过）')
    parser.add_argument('--subtitle_root', type=str, default=os.path.join(LVB_ROOT, 'subtitles'), help='字幕文件的根目录')
    # 新增参数
    parser.add_argument('--sim_threshold', type=float, default=0.5,
                        help='文本相似度提升阈值，高于此阈值的相似度将被放大')
    parser.add_argument('--time_window', type=float, default=2.0,
                        help='文本匹配的时间窗口大小（秒），用于扩展字幕影响范围')
    parser.add_argument('--debug_mode', action='store_true',
                        help='启用调试模式，打印更多日志信息')
    parser.add_argument('--analyze_results', action='store_true',
                        help='执行结果分析，比较关键帧与字幕匹配情况')
    parser.add_argument('--retain_sampling_history', action='store_true',
                        help='在输出结果中保留搜索过程中每轮迭代的采样概率分布')
    parser.add_argument('--run_name', type=str, default='', help='附加到 run_id 的标签（如 smoke / full）')
    parser.add_argument('--wandb_project', type=str, default='vsi-lvb-rerun', help='wandb 项目名')
    parser.add_argument('--wandb_entity', type=str, default='', help='wandb entity（默认账号）')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='online 失败时自动回退 offline')
    parser.add_argument('--wandb_tags', type=str, default='', help='逗号分隔的 wandb tags')
    args = parser.parse_args()

    # 如果输出目录不存在则创建
    os.makedirs(args.output_dir, exist_ok=True)

    # 实验记录：独立 run 目录 + 命令/配置/环境/代码快照/日志 + wandb
    recorder = ExperimentRecorder(args, input_files={
        "grounding_json": os.path.join(args.grounding_dir, args.data_path),
        "yolo_config": args.config_path,
        "yolo_checkpoint": args.checkpoint_path,
        "video_root": args.video_root,
        "subtitle_root": args.subtitle_root,
    })
    run_status = "failed"
    results, output_json = [], None

    # 设置调试日志
    debug_logger = setup_debug_logging(args)
    
    try:
        # 初始化目标定位器
        # target/cue objects 直接读 grounding JSON，搜索过程不调用 LLM，因此不初始化 grounder
        grounder = None

        # 初始化YOLO接口
        yolo_interface = YoloWorldInterface(
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            device=args.device
        )

        input_root = args.grounding_dir
        output_root = recorder.run_dir  # 结果与本次运行的配置/日志放在一起
        os.makedirs(output_root, exist_ok=True)
        results = []
        
        # 添加调试模式下的统计变量
        if args.debug_mode:
            videos_with_matches = 0  # 统计匹配数>=1的视频数量
            total_videos_with_position = 0  # 统计有position信息的视频总数
            total_iterations = 0
            total_search_time = 0.0
            total_total_time = 0.0
            processed_count = 0

        output_json = os.path.join(output_root, args.data_path.replace("obj_", "MVSLS_" + str(args.num) + "_F" +str(args.search_nframes) + "_w" + str(args.text_weight) + "_rel_none" + "_"))

        # 从之前的object_grounding结果加载
        with open(os.path.join(input_root, args.data_path), 'r', encoding='utf-8') as f_read:
            dataset = json.load(f_read)
            print(f"Loaded {len(dataset)} items from previous results")
        # 原 JSON 里的 video_path 指向旧服务器，按文件名重定向到本机视频目录
        for item in dataset:
            item['video_path'] = os.path.join(args.video_root, os.path.basename(item['video_path']))
        if args.filter_subtitle:
            dataset = [item for item in dataset if 'subtitle' not in item['question'].lower()]

        if args.num > 0:
            dataset = dataset[:args.num]

        for idx, data_item in enumerate(dataset):
            print(f"Processing {idx+1}/{len(dataset)}: {data_item['video_id']}")
            
            # 检查是否为字幕相关问题
            is_subtitle_question = 'subtitle' in data_item['question'].lower() or 'caption' in data_item['question'].lower() or 'mention' in data_item['question'].lower()
            if is_subtitle_question and args.debug_mode:
                print(f"发现字幕相关问题: {data_item['question']}")
                
            try:
                # 复制一份参数
                custom_args = copy.deepcopy(args)
                
                # 使用定制参数处理视频
                result = process_TStar_onVideo_multimodal(custom_args, data_item=data_item, grounder=grounder, yolo_scorer=yolo_interface)
                print(f"Completed: {data_item['video_id']}\n")
                
                # 调试模式下统计匹配视频数量
                if args.debug_mode and 'position' in data_item and data_item['position']:
                    total_videos_with_position += 1
                    # 检查是否有匹配分析结果
                    if 'position_match_analysis' in result and result['position_match_analysis'].get('matches', 0) >= 1:
                        videos_with_matches += 1
                        print(f"视频 {data_item['video_id']} 有 {result['position_match_analysis']['matches']} 个匹配帧!")
                # 统计迭代和耗时
                if args.debug_mode and "num_iterations" in result:
                    total_iterations += result["num_iterations"]
                    total_search_time += result.get("search_time", 0)
                    total_total_time += result.get("total_time", 0)
                    processed_count += 1

            except Exception as e:
                print(f"Error processing {data_item['video_id']}: {e}")
                result = {
                    "video_id": data_item.get('video_id', ''),
                    "grounding_objects": [],
                    "keyframe_timestamps": [],
                    "answer": "",
                    "error": str(e)
                }
            
            # 保存结果中添加是否为字幕问题的标记
            result["is_subtitle_question"] = is_subtitle_question
            data_item.update(result)
            results.append(data_item)
            recorder.log_item(idx, data_item, result)
            if (idx + 1) % args.save_batch == 0 or (idx + 1) == len(dataset):
                # 保存批量结果到output_json
                with open(output_json, 'w', encoding='utf-8') as f_out:
                    json.dump(results, f_out, indent=4, ensure_ascii=False)
        
        # 输出调试模式下的统计信息
        if args.debug_mode and total_videos_with_position > 0:
            print("\n===== 关键帧匹配统计 =====")
            print(f"总计有 {total_videos_with_position} 个视频包含position信息")
            print(f"其中 {videos_with_matches} 个视频有至少1个关键帧与Ground Truth匹配")
            match_rate = videos_with_matches / total_videos_with_position * 100
            print(f"匹配视频占比: {match_rate:.2f}%")
        # 输出平均迭代和耗时统计
        if args.debug_mode and processed_count > 0:
            print("\n===== 搜索统计信息 =====")
            print(f"平均每轮迭代次数: {total_iterations / processed_count:.2f}")
            print(f"平均perform_search搜索时长: {total_search_time / processed_count:.2f} 秒")
            print(f"平均单视频总耗时: {total_total_time / processed_count:.2f} 秒")

        # 执行结果分析
        if args.analyze_results:
            analyze_results(results, args)

        print(f"批量处理完成. 结果已保存到 {output_json}")
        run_status = "finished"
        
    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保在程序结束时关闭日志文件并恢复stdout
        if debug_logger:
            debug_logger.close()
            print(f"调试日志已保存完成")
        recorder.finish(results, output_json, status=run_status)


if __name__ == "__main__":
    main()