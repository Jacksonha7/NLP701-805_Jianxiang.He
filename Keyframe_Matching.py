#!/usr/bin/env python3
"""
关键帧匹配率计算脚本

该脚本用于计算关键帧与Ground Truth位置的匹配率。
当设置num_frame=k时，会取top-k个关键帧时间戳与position中的ground truth帧进行对比。

使用方法:
    python Keyframe_Matching.py --input_json path/to/results.json --num_frame 4 --threshold 200
"""

import os
import json
import argparse
import numpy as np
from decord import VideoReader, cpu
from typing import List, Dict, Tuple
import datetime


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="关键帧匹配率计算工具")
    
    # 必需参数
    parser.add_argument('--input_json', type=str, required=True,
                        help='输入的结果JSON文件路径')
    parser.add_argument('--num_frame', type=int, required=True,
                        help='取前k个关键帧进行匹配计算')
    
    # 可选参数
    parser.add_argument('--threshold', type=int, default=200,
                        help='匹配阈值（帧数），默认200帧')
    parser.add_argument('--output_dir', type=str, default='./matching_results',
                        help='输出目录，默认./matching_results')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，输出详细匹配信息')
    
    return parser.parse_args()


def get_video_fps(video_path: str) -> float:
    """获取视频的FPS"""
    try:
        video_reader = VideoReader(video_path, ctx=cpu(0))
        fps = video_reader.get_avg_fps()
        return fps
    except Exception as e:
        print(f"获取视频FPS失败 {video_path}: {e}")
        return 25.0  # 默认FPS


def analyze_position_type(position: List, fps: float) -> Tuple[List[int], bool]:
    """
    分析position的类型（时间戳还是帧号）并转换为帧号
    
    参数:
        position: position列表
        fps: 视频帧率
        
    返回:
        frame_positions: 转换后的帧位置列表
        is_frame: 原始position是否为帧号
    """
    if not position:
        return [], False
    
    # 检查是否为帧号（假设大于30的值很可能是帧号）
    position_is_frame = all(pos > 30 for pos in position if isinstance(pos, (int, float)))
    
    if position_is_frame:
        # 如果已经是帧号，直接使用
        frame_positions = [int(pos) if isinstance(pos, (int, float)) else None for pos in position]
        return frame_positions, True
    else:
        # 如果是时间戳（秒），转换为帧号
        frame_positions = [int(pos * fps) if isinstance(pos, (int, float)) else None for pos in position]
        return frame_positions, False


def calculate_frame_matches(keyframe_timestamps: List[float], 
                          gt_positions: List[int], 
                          fps: float,
                          threshold: int = 200) -> List[Dict]:
    """
    计算关键帧与Ground Truth的匹配情况
    
    参数:
        keyframe_timestamps: 关键帧时间戳列表
        gt_positions: Ground Truth帧位置列表
        fps: 视频帧率
        threshold: 匹配阈值（帧数）
        
    返回:
        match_results: 匹配结果列表
    """
    # 将时间戳转换为帧号
    keyframe_frames = [int(ts * fps) for ts in keyframe_timestamps]
    
    match_results = []
    
    for i, kf_frame in enumerate(keyframe_frames):
        best_match = {'gt_frame': None, 'distance': float('inf'), 'match': False}
        
        for gt_frame in gt_positions:
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
    
    return match_results


def process_single_item(data_item: Dict, args: argparse.Namespace) -> Dict:
    """
    处理单个数据项的关键帧匹配
    
    参数:
        data_item: 单个数据项
        args: 命令行参数
        
    返回:
        result: 处理结果
    """
    result = {
        'video_id': data_item.get('video_id', ''),
        'video_path': data_item.get('video_path', ''),
        'has_position': False,
        'has_keyframes': False,
        'match_results': [],
        'matches': 0,
        'total': 0,
        'match_rate': 0.0,
        'error': None
    }
    
    try:
        # 检查是否有position信息
        if 'position' not in data_item or not data_item['position']:
            result['error'] = 'No position information'
            return result
        result['has_position'] = True
        
        # 检查是否有关键帧信息
        if 'keyframe_timestamps' not in data_item or not data_item['keyframe_timestamps']:
            result['error'] = 'No keyframe timestamps'
            return result
        result['has_keyframes'] = True
        
        # 获取前num_frame个关键帧
        keyframe_timestamps = data_item['keyframe_timestamps'][:args.num_frame]
        result['total'] = len(keyframe_timestamps)
        
        if result['total'] == 0:
            result['error'] = 'No keyframes after filtering'
            return result
        
        # 获取视频FPS
        fps = get_video_fps(data_item['video_path'])
        
        # 分析position类型并转换
        gt_positions, is_frame = analyze_position_type(data_item['position'], fps)
        
        if args.debug:
            print(f"\n===== 处理视频: {data_item['video_id']} =====")
            print(f"视频路径: {data_item['video_path']}")
            print(f"视频FPS: {fps}")
            print(f"Position类型: {'帧号' if is_frame else '时间戳'}")
            print(f"Ground Truth位置: {data_item['position']}")
            print(f"转换后帧位置: {gt_positions}")
            print(f"关键帧时间戳: {keyframe_timestamps}")
        
        # 计算匹配结果
        match_results = calculate_frame_matches(
            keyframe_timestamps, gt_positions, fps, args.threshold
        )
        result['match_results'] = match_results
        
        # 统计匹配数量
        matches = sum(1 for r in match_results if r['match'])
        result['matches'] = matches
        result['match_rate'] = matches / result['total'] if result['total'] > 0 else 0
        
        if args.debug:
            print(f"\n===== 匹配结果 =====")
            for i, match_result in enumerate(match_results):
                if match_result['best_match_gt'] is not None:
                    match_status = "✓" if match_result['match'] else "✗"
                    print(f"关键帧 #{i+1} ({match_result['keyframe_time']:.2f}s, 第{match_result['keyframe_frame']}帧) -> " + 
                          f"最接近GT: 第{match_result['best_match_gt']}帧, 距离: {match_result['distance']}帧 {match_status}")
            
            print(f"\n匹配结果: {matches}/{result['total']} 关键帧与Ground Truth匹配 " + 
                  f"({result['match_rate']*100:.2f}%)")
            print(f"阈值: {args.threshold} 帧")
        
    except Exception as e:
        result['error'] = str(e)
        if args.debug:
            print(f"处理视频 {data_item.get('video_id', 'unknown')} 时出错: {e}")
    
    return result


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载JSON文件
    print(f"加载JSON文件: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共加载 {len(data)} 个数据项")
    
    # 处理每个数据项
    results = []
    total_videos_with_position = 0
    total_videos_with_keyframes = 0
    total_matches = 0
    total_keyframes = 0
    
    for idx, data_item in enumerate(data):
        if args.debug:
            print(f"\n处理进度: {idx+1}/{len(data)}")
        
        result = process_single_item(data_item, args)
        results.append(result)
        
        # 统计信息
        if result['has_position']:
            total_videos_with_position += 1
        if result['has_keyframes']:
            total_videos_with_keyframes += 1
            total_matches += result['matches']
            total_keyframes += result['total']
    
    # 计算总体统计
    overall_match_rate = total_matches / total_keyframes if total_keyframes > 0 else 0
    videos_with_matches = sum(1 for r in results if r['matches'] > 0 and r['has_position'])
    
    # 输出总体统计
    print(f"\n{'='*60}")
    print(f"关键帧匹配率统计报告")
    print(f"{'='*60}")
    print(f"总视频数: {len(data)}")
    print(f"有position信息的视频数: {total_videos_with_position}")
    print(f"有关键帧信息的视频数: {total_videos_with_keyframes}")
    print(f"有至少1个匹配的视频数: {videos_with_matches}")
    print(f"总关键帧数: {total_keyframes}")
    print(f"总匹配数: {total_matches}")
    print(f"总体匹配率: {videos_with_matches/total_videos_with_keyframes*100:.2f}%")
    print(f"匹配阈值: {args.threshold} 帧")
    print(f"取前 {args.num_frame} 个关键帧")
    
    # 保存详细结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"matching_results_k{args.num_frame}_t{args.threshold}_{timestamp}.json")
    
    output_data = {
        'summary': {
            'total_videos': len(data),
            'videos_with_position': total_videos_with_position,
            'videos_with_keyframes': total_videos_with_keyframes,
            'videos_with_matches': videos_with_matches,
            'total_keyframes': total_keyframes,
            'total_matches': total_matches,
            'overall_match_rate': overall_match_rate,
            'threshold': args.threshold,
            'num_frame': args.num_frame,
            'timestamp': timestamp
        },
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=3, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 输出匹配率分布
    match_rates = [r['match_rate'] for r in results if r['has_position'] and r['has_keyframes']]
    if match_rates:
        print(f"\n匹配率分布:")
        print(f"  0%: {sum(1 for r in match_rates if r == 0)} 个视频")
        print(f"  1-25%: {sum(1 for r in match_rates if 0 < r <= 0.25)} 个视频")
        print(f"  26-50%: {sum(1 for r in match_rates if 0.25 < r <= 0.5)} 个视频")
        print(f"  51-75%: {sum(1 for r in match_rates if 0.5 < r <= 0.75)} 个视频")
        print(f"  76-99%: {sum(1 for r in match_rates if 0.75 < r < 1)} 个视频")
        print(f"  100%: {sum(1 for r in match_rates if r == 1)} 个视频")


if __name__ == "__main__":
    main() 