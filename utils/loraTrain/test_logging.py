#!/usr/bin/env python3
"""
测试LoRA训练完整日志系统
"""
import sys
import os
import logging
import time
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.loraTrain.log import setup_logging, cleanup_logging

def test_logging_system():
    """测试日志系统的各种功能"""
    
    # 创建模拟的args对象
    class MockArgs:
        def __init__(self):
            self.dataset_type = "test"
            self.precision = "bf16"
            self.corpus_path = "/tmp/test_corpus"
            self.model_name = "/tmp/test_model"
            self.loraadaptor_save_path_base = "/tmp/test_lora"
            self.benchmark_paths = ["test1.json", "test2.json"]
            self.benchmark_data_path = "test_benchmark.json"
            self.world_size = 1
            self.rank = 0
    
    args = MockArgs()
    
    print("=" * 60)
    print("开始测试LoRA训练完整日志系统")
    print("=" * 60)
    
    # 设置日志系统
    log_dir = setup_logging(args)
    print(f"日志目录: {log_dir}")
    
    # 测试1: print语句
    print("\n🧪 测试1: print语句输出")
    print("这是一个普通的print语句")
    print("这是包含中文的print语句：你好世界！")
    print("这是包含特殊字符的print语句：@#$%^&*()")
    
    # 测试2: logging输出
    print("\n🧪 测试2: logging模块输出")
    logging.debug("这是debug级别的日志")
    logging.info("这是info级别的日志")
    logging.warning("这是warning级别的日志")
    logging.error("这是error级别的日志")
    
    # 测试3: 多行输出
    print("\n🧪 测试3: 多行输出")
    multiline_text = """这是多行文本测试：
第一行
第二行
第三行
结束"""
    print(multiline_text)
    
    # 测试4: 格式化输出
    print("\n🧪 测试4: 格式化输出")
    for i in range(3):
        print(f"循环输出 {i+1}: 当前时间 {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(0.5)
    
    # 测试5: 标准错误输出
    print("\n🧪 测试5: 标准错误输出")
    sys.stderr.write("这是写入到stderr的信息\n")
    sys.stderr.flush()
    
    # 测试6: 异常处理
    print("\n🧪 测试6: 异常处理")
    try:
        # 故意引发异常
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"捕获到异常: {e}")
        logging.error(f"异常详情: {e}")
    
    # 测试7: 长时间运行模拟
    print("\n🧪 测试7: 模拟训练过程")
    packages = ["numpy", "pandas", "torch", "transformers"]
    for i, pkg in enumerate(packages):
        print(f"正在处理包 {pkg} ({i+1}/{len(packages)})...")
        logging.info(f"开始训练 {pkg} 包的LoRA模型")
        time.sleep(1)
        print(f"✅ 包 {pkg} 训练完成")
        logging.info(f"✅ {pkg} 训练统计: 成功=1, 跳过=0, 失败=0")
    
    # 测试8: 大量输出
    print("\n🧪 测试8: 大量输出测试")
    for i in range(20):
        if i % 5 == 0:
            logging.info(f"批次 {i//5 + 1} 开始处理")
        print(f"处理项目 {i+1}: {'■' * (i % 10 + 1)}")
    
    # 测试9: Unicode和特殊字符
    print("\n🧪 测试9: Unicode和特殊字符")
    print("emoji测试: 🚀 🎯 ✅ ❌ 📝 💾")
    print("Unicode测试: αβγδε Ñiño 测试")
    print("特殊字符: <>?:\"|{}+_)(*&^%$#@!~`")
    
    # 测试10: 模拟错误场景
    print("\n🧪 测试10: 模拟错误场景")
    logging.warning("这是一个警告消息")
    logging.error("这是一个错误消息")
    print("ERROR: 模拟的错误信息")
    print("FAILED: 模拟的失败信息")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    
    # 显示日志文件信息
    log_file = os.path.join(log_dir, "train_lora_test_complete.log")
    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file)
        print(f"📝 日志文件: {log_file}")
        print(f"📊 文件大小: {file_size} bytes")
        
        # 显示日志文件的最后几行
        print("\n📋 日志文件最后10行:")
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"读取日志文件失败: {e}")
    
    return log_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试LoRA训练完整日志系统")
    parser.add_argument("--cleanup", action="store_true", help="测试完成后手动清理日志系统")
    
    args = parser.parse_args()
    
    try:
        log_dir = test_logging_system()
        
        if args.cleanup:
            print("\n🧹 手动清理日志系统...")
            cleanup_logging()
            print("✅ 清理完成")
        else:
            print("\n💡 提示: 程序退出时会自动清理日志系统")
            print("💡 如需手动清理，请使用 --cleanup 参数")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        cleanup_logging()
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        cleanup_logging()
        raise

if __name__ == "__main__":
    main() 