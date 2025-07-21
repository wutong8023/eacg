#!/usr/bin/env python3
"""
数据分布正交性检查器
用于在分布式训练过程中实时检查数据分布的正交性
"""

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Set, Tuple, Optional
import logging
from collections import defaultdict
import json
import os
from datetime import datetime
import math


class OrthogonalDistributedSampler(DistributedSampler):
    """
    确保正交性和数据量一致的分布式采样器
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        # 重新计算样本数量，确保所有rank获得相同数量的样本
        total_size = len(dataset)
        
        if drop_last:
            # 丢弃不能被均匀分配的样本，确保每个rank获得相同数量
            samples_per_rank = total_size // self.num_replicas
            self.num_samples = samples_per_rank
            self.total_size = samples_per_rank * self.num_replicas
        else:
            # 如果不丢弃，则进行padding（不推荐用于分布式训练）
            samples_per_rank = math.ceil(total_size / self.num_replicas)
            self.num_samples = samples_per_rank
            self.total_size = samples_per_rank * self.num_replicas
        
        logging.info(f"Rank {self.rank}: OrthogonalDistributedSampler - "
                    f"原始数据集大小: {total_size}, "
                    f"每个rank样本数: {self.num_samples}, "
                    f"总使用样本数: {self.total_size}")
    
    def __iter__(self):
        if self.shuffle:
            # 使用固定种子确保所有rank生成相同的shuffle顺序
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # 确保数据分配是连续且不重叠的
        if self.drop_last:
            # 只使用能被均匀分配的数据
            indices = indices[:self.total_size]
        else:
            # 如果数据不够，进行重复填充（不推荐）
            if len(indices) < self.total_size:
                indices += indices[:self.total_size - len(indices)]
        
        # 为当前rank分配连续的数据块
        indices = indices[self.rank::self.num_replicas]
        
        # 确保返回正确数量的样本
        assert len(indices) == self.num_samples, f"Rank {self.rank}: 期望 {self.num_samples} 个样本，实际得到 {len(indices)} 个"
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class DataDistributionChecker:
    """数据分布正交性检查器"""
    
    def __init__(self, rank: int, world_size: int, enable_checking: bool = True):
        """
        初始化检查器
        
        Args:
            rank: 当前进程rank
            world_size: 总进程数
            enable_checking: 是否启用检查
        """
        self.rank = rank
        self.world_size = world_size
        self.enable_checking = enable_checking
        self.check_results = {
            'epochs': [],
            'total_overlaps': 0,
            'total_coverage_issues': 0,
            'orthogonal_epochs': 0,
            'complete_coverage_epochs': 0,
            'equal_distribution_epochs': 0
        }
        
        if rank == 0:
            logging.info(f"数据分布检查器初始化完成 - World Size: {world_size}")
    
    def collect_indices_from_all_ranks(self, indices: List[int]) -> Optional[List[List[int]]]:
        """
        收集所有rank的索引列表
        
        Args:
            indices: 当前rank的索引列表
        
        Returns:
            所有rank的索引列表（仅在rank 0上返回完整数据）
        """
        if not self.enable_checking:
            return None
            
        # 准备收集容器
        all_indices = [None] * self.world_size
        
        # 使用gather收集所有rank的索引
        dist.gather_object(
            indices,
            all_indices if self.rank == 0 else None,
            dst=0
        )
        
        return all_indices if self.rank == 0 else None
    
    def check_orthogonality_and_distribution(self, all_indices: List[List[int]], epoch: int, batch_count: int = 0) -> Dict:
        """
        检查数据分布的正交性和均匀性
        
        Args:
            all_indices: 所有rank的索引列表
            epoch: 当前epoch
            batch_count: 当前batch计数
        
        Returns:
            检查结果字典
        """
        if not self.enable_checking or self.rank != 0:
            return {}
        
        result = {
            'epoch': epoch,
            'batch_count': batch_count,
            'orthogonal': True,
            'equal_distribution': True,
            'overlaps': [],
            'coverage': {},
            'rank_counts': [],
            'distribution_stats': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查每个rank的数据量
        rank_counts = []
        for i, indices in enumerate(all_indices):
            if indices is not None:
                rank_counts.append(len(indices))
            else:
                rank_counts.append(0)
        
        result['rank_counts'] = rank_counts
        
        # 检查数据量是否相等
        if len(set(rank_counts)) > 1:
            result['equal_distribution'] = False
            result['distribution_stats'] = {
                'min_count': min(rank_counts),
                'max_count': max(rank_counts),
                'mean_count': sum(rank_counts) / len(rank_counts),
                'std_dev': ((sum((x - sum(rank_counts)/len(rank_counts))**2 for x in rank_counts) / len(rank_counts))**0.5)
            }
            
            if self.rank == 0:
                logging.warning(f"Epoch {epoch}, Batch {batch_count}: 数据分布不均 - 各rank数据量: {rank_counts}")
                logging.warning(f"  最小: {result['distribution_stats']['min_count']}, "
                              f"最大: {result['distribution_stats']['max_count']}, "
                              f"标准差: {result['distribution_stats']['std_dev']:.2f}")
        else:
            result['distribution_stats'] = {
                'uniform_count': rank_counts[0] if rank_counts else 0,
                'all_equal': True
            }
            
            if self.rank == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_count}: ✅ 数据分布均匀 - 每个rank: {rank_counts[0]} 个样本")
        
        # 检查正交性（无重叠）
        overlap_found = False
        for i in range(self.world_size):
            for j in range(i + 1, self.world_size):
                if all_indices[i] is not None and all_indices[j] is not None:
                    set_i = set(all_indices[i])
                    set_j = set(all_indices[j])
                    overlap = set_i & set_j
                    
                    if overlap:
                        overlap_info = {
                            'rank1': i,
                            'rank2': j,
                            'overlap_count': len(overlap),
                            'overlap_indices': list(overlap)[:5]  # 只显示前5个
                        }
                        result['overlaps'].append(overlap_info)
                        result['orthogonal'] = False
                        overlap_found = True
                        
                        if self.rank == 0:
                            logging.error(f"Epoch {epoch}, Batch {batch_count}: ❌ Rank {i}和{j}有{len(overlap)}个重叠索引")
                            if len(overlap) <= 5:
                                logging.error(f"  重叠索引: {list(overlap)}")
                            else:
                                logging.error(f"  重叠索引示例: {list(overlap)[:5]}...")
        
        if not overlap_found and self.rank == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_count}: ✅ 数据正交性检查通过")
        
        # 检查覆盖率
        total_indices = set()
        for indices in all_indices:
            if indices is not None:
                total_indices.update(indices)
        
        expected_total = sum(rank_counts)
        actual_unique = len(total_indices)
        
        result['coverage'] = {
            'expected_total': expected_total,
            'actual_unique': actual_unique,
            'coverage_ratio': actual_unique / expected_total if expected_total > 0 else 0,
            'complete': actual_unique == expected_total
        }
        
        if not result['coverage']['complete'] and self.rank == 0:
            missing = expected_total - actual_unique
            logging.warning(f"Epoch {epoch}, Batch {batch_count}: ⚠️  数据覆盖不完整 - 重复数据: {missing}个")
        elif self.rank == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_count}: ✅ 数据覆盖完整 ({actual_unique}/{expected_total})")
        
        return result
    
    def check_sampler_indices(self, sampler, epoch: int, batch_count: int = 0) -> Dict:
        """
        检查采样器的索引分布
        
        Args:
            sampler: 分布式采样器
            epoch: 当前epoch
            batch_count: 当前batch计数
        
        Returns:
            检查结果
        """
        if not self.enable_checking:
            return {}
        
        # 收集当前rank的索引
        indices = list(sampler)
        
        # 打印当前rank的索引信息
        if len(indices) > 0:
            print(f"Rank {self.rank}, Epoch {epoch}: 索引范围 {min(indices)}-{max(indices)}, 数量: {len(indices)}")
            if len(indices) <= 10:
                print(f"  完整索引: {indices}")
            else:
                print(f"  索引示例: {indices[:5]}...{indices[-5:]}")
        else:
            print(f"Rank {self.rank}, Epoch {epoch}: 无索引数据")
        
        # 收集所有rank的索引
        all_indices = self.collect_indices_from_all_ranks(indices)
        
        # 检查正交性和分布
        result = self.check_orthogonality_and_distribution(all_indices, epoch, batch_count)
        
        # 记录结果
        if result and self.rank == 0:
            self.check_results['epochs'].append(result)
            
            if not result['orthogonal']:
                self.check_results['total_overlaps'] += len(result['overlaps'])
            else:
                self.check_results['orthogonal_epochs'] += 1
            
            if result['coverage'].get('complete', False):
                self.check_results['complete_coverage_epochs'] += 1
            else:
                self.check_results['total_coverage_issues'] += 1
                
            if result['equal_distribution']:
                self.check_results['equal_distribution_epochs'] += 1
        
        return result
    
    def print_summary(self):
        """打印检查总结"""
        if not self.enable_checking or self.rank != 0:
            return
        
        print("\n" + "="*80)
        print("数据分布正交性和均匀性检查总结")
        print("="*80)
        
        total_checks = len(self.check_results['epochs'])
        if total_checks == 0:
            print("未进行任何检查")
            return
            
        orthogonal_checks = self.check_results['orthogonal_epochs']
        coverage_checks = self.check_results['complete_coverage_epochs']
        equal_dist_checks = self.check_results['equal_distribution_epochs']
        
        print(f"总检查次数: {total_checks}")
        print(f"正交性检查通过: {orthogonal_checks}/{total_checks} ({orthogonal_checks/total_checks*100:.1f}%)")
        print(f"覆盖率检查通过: {coverage_checks}/{total_checks} ({coverage_checks/total_checks*100:.1f}%)")
        print(f"均匀分布检查通过: {equal_dist_checks}/{total_checks} ({equal_dist_checks/total_checks*100:.1f}%)")
        print(f"发现重叠次数: {self.check_results['total_overlaps']}")
        print(f"覆盖率问题次数: {self.check_results['total_coverage_issues']}")
        
        # 显示详细问题
        if self.check_results['total_overlaps'] > 0:
            print("\n❌ 重叠详情:")
            for epoch_result in self.check_results['epochs']:
                if not epoch_result['orthogonal']:
                    print(f"  Epoch {epoch_result['epoch']}, Batch {epoch_result['batch_count']}: {len(epoch_result['overlaps'])} 个重叠")
        
        if self.check_results['total_coverage_issues'] > 0:
            print("\n⚠️  覆盖率问题详情:")
            for epoch_result in self.check_results['epochs']:
                if not epoch_result['coverage'].get('complete', True):
                    missing = epoch_result['coverage']['expected_total'] - epoch_result['coverage']['actual_unique']
                    print(f"  Epoch {epoch_result['epoch']}, Batch {epoch_result['batch_count']}: 重复 {missing} 个数据")
        
        # 显示分布不均问题
        unequal_dist_count = total_checks - equal_dist_checks
        if unequal_dist_count > 0:
            print(f"\n⚠️  数据分布不均次数: {unequal_dist_count}")
            for epoch_result in self.check_results['epochs']:
                if not epoch_result['equal_distribution']:
                    stats = epoch_result['distribution_stats']
                    print(f"  Epoch {epoch_result['epoch']}: 数据量范围 {stats['min_count']}-{stats['max_count']}, 标准差: {stats['std_dev']:.2f}")
        
        print("="*80)
    
    def save_results(self, output_file: str = "data_distribution_check_results.json"):
        """保存检查结果到文件"""
        if not self.enable_checking or self.rank != 0:
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.check_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"数据分布检查结果已保存到: {output_file}")


def create_distributed_dataloader_with_checking(dataset, batch_size, rank, world_size, shuffle=True, enable_checking=True):
    """
    创建带检查功能的分布式数据加载器，确保数据正交且均匀分布
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        rank: 当前rank
        world_size: 总进程数
        shuffle: 是否打乱
        enable_checking: 是否启用数据分布检查
    
    Returns:
        (dataloader, checker): 数据加载器和检查器
    """
    from utils.loraTrain.buildandloadData import collate_fn
    
    # 创建检查器
    checker = DataDistributionChecker(rank, world_size, enable_checking)
    
    if world_size > 1:
        # 打印数据集信息
        if rank == 0:
            print(f"数据集原始大小: {len(dataset)}")
            print(f"World size: {world_size}")
            print(f"预期每rank数据量: {len(dataset) // world_size}")
        
        # 使用改进的正交分布式采样器
        sampler = OrthogonalDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=42,  # 确保可重复性
            drop_last=True  # 确保数据量一致
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer if hasattr(dataset, 'tokenizer') else None),
            pin_memory=True,
            num_workers=2,
            drop_last=True  # 再次确保batch大小一致
        )
        
        # 打印详细的数据分配信息
        num_samples = len(sampler)
        num_batches = len(dataloader)
        print(f"Rank {rank}: 实际分配 {num_samples} 个样本, {num_batches} 个batch")
        
        # 收集并验证所有rank的数据量
        sample_counts = torch.tensor([num_samples], dtype=torch.long).cuda()
        batch_counts = torch.tensor([num_batches], dtype=torch.long).cuda()
        
        # 收集所有rank的计数
        all_sample_counts = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
        all_batch_counts = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
        
        dist.all_gather(all_sample_counts, sample_counts)
        dist.all_gather(all_batch_counts, batch_counts)
        
        if rank == 0:
            # 从张量列表中提取数值
            sample_counts_list = [tensor.cpu().item() for tensor in all_sample_counts]
            batch_counts_list = [tensor.cpu().item() for tensor in all_batch_counts]
            
            print(f"各rank样本数量: {sample_counts_list}")
            print(f"各rank batch数量: {batch_counts_list}")
            
            # 检查是否一致
            if len(set(sample_counts_list)) == 1:
                print(f"✅ 样本分布均匀: 每个rank {sample_counts_list[0]} 个样本")
            else:
                print(f"❌ 样本分布不均: {sample_counts_list}")
                
            if len(set(batch_counts_list)) == 1:
                print(f"✅ Batch分布均匀: 每个rank {batch_counts_list[0]} 个batch")
            else:
                print(f"❌ Batch分布不均: {batch_counts_list}")
            
            print(f"总样本数: {sum(sample_counts_list)}, 总batch数: {sum(batch_counts_list)}")
            
        # 初始检查采样器索引分布
        if enable_checking:
            print(f"Rank {rank}: 执行初始数据分布检查...")
            checker.check_sampler_indices(sampler, epoch=0, batch_count=0)
            
    else:
        # 单GPU训练
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer if hasattr(dataset, 'tokenizer') else None),
            pin_memory=True,
            num_workers=2,
            drop_last=True
        )
        
        print(f"单GPU模式: {len(dataloader)} 个batch")
    
    return dataloader, checker


def check_batch_distribution(dataloader, checker, epoch: int, check_interval: int = 10):
    """
    在训练循环中检查batch分布
    
    Args:
        dataloader: 数据加载器
        checker: 数据分布检查器
        epoch: 当前epoch
        check_interval: 检查间隔（每隔多少个batch检查一次）
    """
    if not checker or not checker.enable_checking:
        for batch in dataloader:
            yield batch
        return
    
    batch_count = 0
    
    for batch in dataloader:
        batch_count += 1
        
        # 每隔check_interval个batch检查一次（这里主要是输出信息，真正的检查在epoch级别）
        if batch_count % check_interval == 0 and checker.rank == 0:
            print(f"Rank {checker.rank}, Epoch {epoch}, Batch {batch_count}/{len(dataloader)} - 数据分布检查正常运行")
        
        yield batch 