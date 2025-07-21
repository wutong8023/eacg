import torch
import logging
def log_detailed_gpu_memory_report(rank=0, stage=""):
    """
    记录并打印详细的GPU内存使用报告.
    """
    if rank == 0 and torch.cuda.is_available():
        logging.info(f"--- Detailed GPU Memory Report [{stage}] ---")
        print(f"--- Detailed GPU Memory Report [{stage}] ---")
        try:
            # 确保所有CUDA操作完成
            torch.cuda.synchronize()
            # 打印每个GPU的摘要
            for i in range(torch.cuda.device_count()):
                logging.info(f"--- Report for GPU {i} ---")
                print(f"--- Report for GPU {i} ---")
                summary = torch.cuda.memory_summary(device=i, abbreviated=False)
                logging.info(summary)
                print(summary)
        except Exception as e:
            logging.error(f"生成详细内存报告时出错: {e}")
        logging.info("--- End of Detailed GPU Memory Report ---")
        print("--- End of Detailed GPU Memory Report ---")


def find_and_clear_lingering_tensors(rank=0, stage=""):
    """
    查找并报告可能残留的GPU张量，帮助诊断内存泄漏.
    """
    if rank == 0:
        logging.info(f"--- Searching for Lingering GPU Tensors [{stage}] ---")
        print(f"--- Searching for Lingering GPU Tensors [{stage}] ---")
        
        import gc
        leaked_tensors_found = False
        # 强制垃圾回收，以便get_objects能看到所有可回收对象
        gc.collect()
        torch.cuda.empty_cache()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    leaked_tensors_found = True
                    # 打印张量信息
                    info = f"  - Lingering Tensor Found: shape={obj.shape}, dtype={obj.dtype}, size={obj.numel() * obj.element_size() / 1024**2:.2f} MB, device={obj.device}"
                    logging.warning(info)
                    print(info)
                    # 打印其引用者，帮助调试
                    for referrer in gc.get_referrers(obj):
                        # 避免打印出 gc.get_objects() 本身
                        if isinstance(referrer, list) and len(referrer) > 1000: continue
                        referrer_info = f"    - Referred by: {type(referrer)}"
                        logging.warning(referrer_info)
                        print(referrer_info)
            except Exception:
                # 忽略检查中可能出现的错误
                continue
        
        if not leaked_tensors_found:
            msg = "✅ No lingering GPU tensors found by gc."
            logging.info(msg)
            print(msg)
        
        logging.info("--- End of Lingering Tensor Search ---")
        print("--- End of Lingering Tensor Search ---")
def log_gpu_memory_usage(rank=0, stage="", package_info=""):
    """记录GPU内存使用情况"""
    if rank == 0 and torch.cuda.is_available():
        logging.info(f"=== GPU内存使用情况 [{stage}] {package_info} ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            free = total - cached
            logging.info(f"GPU {i}: 已分配={allocated:.2f}GB, 已缓存={cached:.2f}GB, 可用={free:.2f}GB, 总计={total:.2f}GB")
        logging.info("=" * 50)


def clear_model_outputs_and_cache(rank=0, stage=""):
    """清理模型输出对象和attention缓存"""
    if rank == 0:
        logging.info(f"清理模型输出对象和缓存 [{stage}]")
    
    import gc
    
    # 查找并清理transformers模型输出对象
    cleared_outputs = 0
    cleared_tensors = 0
    
    for obj in gc.get_objects():
        try:
            # 清理CausalLMOutputWithPast对象
            if hasattr(obj, '__class__') and 'CausalLMOutputWithPast' in str(obj.__class__):
                if hasattr(obj, 'logits'):
                    obj.logits = None
                if hasattr(obj, 'past_key_values'):
                    obj.past_key_values = None
                if hasattr(obj, 'hidden_states'):
                    obj.hidden_states = None
                if hasattr(obj, 'attentions'):
                    obj.attentions = None
                cleared_outputs += 1
            
            # 清理残留的CUDA张量
            elif torch.is_tensor(obj) and obj.is_cuda:
                # 特别处理大的logits张量
                if obj.numel() > 1000000:  # 大于1M元素的张量
                    obj.data = torch.empty(0, device='cpu')
                    cleared_tensors += 1
        except Exception:
            continue
    
    if rank == 0:
        logging.info(f"清理了 {cleared_outputs} 个模型输出对象, {cleared_tensors} 个大张量")


def clear_optimizer_states(optimizer=None, rank=0):
    """彻底清理优化器状态"""
    if optimizer is None:
        return
    
    if rank == 0:
        logging.info("清理优化器状态...")
    
    try:
        # 清理优化器状态字典
        if hasattr(optimizer, 'state_dict'):
            state_dict = optimizer.state_dict()
            if 'state' in state_dict:
                state_dict['state'].clear()
            if 'param_groups' in state_dict:
                for group in state_dict['param_groups']:
                    if 'params' in group:
                        group['params'].clear()
        
        # 清理优化器内部状态
        if hasattr(optimizer, 'state'):
            optimizer.state.clear()
        
        # 清理参数组
        if hasattr(optimizer, 'param_groups'):
            for group in optimizer.param_groups:
                if 'params' in group:
                    group['params'].clear()
    except Exception as e:
        if rank == 0:
            logging.warning(f"清理优化器状态时出错: {e}")


def force_clear_cuda_context(rank=0, stage=""):
    """强制清理CUDA上下文中的残留对象"""
    if rank == 0:
        logging.info(f"强制清理CUDA上下文 [{stage}]")
    
    import gc
    
    # 多轮垃圾回收
    for i in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 强制清理CUDA IPC
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def comprehensive_memory_cleanup(rank=0, stage="", package_info=""):
    """综合内存清理函数"""
    if rank == 0:
        logging.info(f"开始综合内存清理 [{stage}] {package_info}")
    
    # 1. 清理模型输出和缓存
    clear_model_outputs_and_cache(rank, stage)
    
    # 2. 强制清理CUDA上下文
    force_clear_cuda_context(rank, stage)
    
    # 3. 记录清理后的内存状态
    log_gpu_memory_usage(rank, f"{stage}_after_comprehensive_cleanup", package_info)


def force_memory_reset_device(rank=0, stage="", package_info=""):
    """使用reset_device强制重置GPU状态"""
    if not torch.cuda.is_available():
        return
    
    torch.cuda.empty_cache()
    # 记录重置后的内存状态
    log_gpu_memory_usage(rank, f"{stage}_after_device_reset", package_info)


# def cleanup_model_and_optimizer(model=None, optimizer=None, rank=0):
#     """安全地删除模型和优化器"""
#     import gc
    
#     if rank == 0:
#         logging.info("清理模型和优化器...")
    
#     # 清理优化器
#     if optimizer is not None:
#         clear_optimizer_states(optimizer, rank)
#         del optimizer
    
#     # 清理模型
#     if model is not None and isinstance(model, torch.nn.Module):
#         try:
#             # 提取DDP包装的模型
#             if hasattr(model, 'module'):
#                 model = model.module
            
#             # 清理模型参数的梯度
#             for param in model.parameters():
#                 param.grad = None
#                 param.data = param.data.cpu()
            
#             # 将模型移到CPU
#             model.to('cpu')
#             del model
#         except Exception as e:
#             if rank == 0:
#                 logging.warning(f"清理模型时出错: {e}")
    
#     # 强制垃圾回收
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     if rank == 0:
#         log_gpu_memory_usage(rank, "model_optimizer_cleanup_complete", "")


# 保持原有的force_cleanup_memory函数，但简化其功能
def force_cleanup_memory(rank=0, stage="", package_info=""):
    """彻底清理CUDA缓存并记录结果"""
    if rank == 0:
        logging.info(f"执行基础内存清理 [{stage}] {package_info}")
    
    # 执行综合清理
    comprehensive_memory_cleanup(rank, stage, package_info)
