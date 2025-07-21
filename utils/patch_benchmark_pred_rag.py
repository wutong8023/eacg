#!/usr/bin/env python3
"""
自动修复benchmark/pred_rag.py中的多进程JSON读写错误
"""

import os
import re
import shutil
import sys
from datetime import datetime

def backup_file(filepath):
    """创建文件备份"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ 已创建备份文件: {backup_path}")
    return backup_path

def apply_fix_to_pred_rag(filepath="benchmark/pred_rag.py"):
    """应用修复到pred_rag.py文件"""
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    print(f"🔧 开始修复文件: {filepath}")
    
    # 创建备份
    backup_path = backup_file(filepath)
    
    try:
        # 读取原文件
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经导入了修复模块
        if 'from multiprocess_utils import' in content:
            print("⚠️  文件似乎已经修复过了，跳过修复")
            return True
        
        # 添加导入语句
        import_pattern = r'(import json\s*\n)'
        import_replacement = r'\1import sys\nimport os\nsys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))\nfrom multiprocess_utils import safe_read_coordination_data, safe_write_coordination_data\n'
        
        if re.search(import_pattern, content):
            content = re.sub(import_pattern, import_replacement, content, count=1)
            print("✅ 已添加安全文件操作导入")
        else:
            # 如果找不到import json，在文件开头添加
            content = '''import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from multiprocess_utils import safe_read_coordination_data, safe_write_coordination_data

''' + content
            print("✅ 在文件开头添加了导入语句")
        
        # 替换有问题的代码段
        problematic_pattern = r'''(\s+)# 读取coordination数据\s*
\s+try:\s*
\s+with open\(unprocessed_data_file, 'r'\) as f:\s*
\s+coordination_data = json\.load\(f\)\s*
\s+
\s+# 重构unprocessed_data\s*
\s+unprocessed_data = \[\]\s*
\s+for item_list in coordination_data\["unprocessed_data"\]:\s*
\s+# 将list转换回tuple\s*
\s+unprocessed_data\.append\(tuple\(item_list\)\)\s*
\s+
\s+logging\.info\(f"Worker {rank}: Loaded coordination data from rank 0"\)\s*
\s+logging\.info\(f"Worker {rank}: Total unprocessed samples: {len\(unprocessed_data\)}"\)'''
        
        replacement = r'''\1# 读取coordination数据 - 使用安全方法
\1try:
\1    coordination_data = safe_read_coordination_data(unprocessed_data_file)
\1    
\1    if not coordination_data or "unprocessed_data" not in coordination_data:
\1        logging.warning(f"Worker {rank}: No valid coordination data found")
\1        unprocessed_data = []
\1    else:
\1        # 重构unprocessed_data
\1        unprocessed_data = []
\1        for item_list in coordination_data["unprocessed_data"]:
\1            if isinstance(item_list, (list, tuple)):
\1                # 将list转换回tuple
\1                unprocessed_data.append(tuple(item_list))
\1            else:
\1                logging.warning(f"Worker {rank}: Skipping invalid item: {item_list}")
\1        
\1        logging.info(f"Worker {rank}: Loaded coordination data from rank 0 safely")
\1        logging.info(f"Worker {rank}: Total unprocessed samples: {len(unprocessed_data)}")'''
        
        if re.search(problematic_pattern, content, re.MULTILINE):
            content = re.sub(problematic_pattern, replacement, content, flags=re.MULTILINE)
            print("✅ 已替换有问题的coordination数据读取代码")
        else:
            # 如果找不到完整模式，尝试更简单的替换
            simple_pattern = r'(\s+)with open\(unprocessed_data_file, \'r\'\) as f:\s*\n\s+coordination_data = json\.load\(f\)'
            simple_replacement = r'\1coordination_data = safe_read_coordination_data(unprocessed_data_file)\n\1if not coordination_data:\n\1    coordination_data = {"unprocessed_data": []}'
            
            if re.search(simple_pattern, content):
                content = re.sub(simple_pattern, simple_replacement, content)
                print("✅ 已使用简单模式替换JSON读取代码")
            else:
                print("⚠️  未找到需要替换的代码模式，请手动修复")
        
        # 查找并替换写入操作（如果存在）
        write_pattern = r'(\s+)with open\([^,]+,\s*[\'"]w[\'"].*?\) as f:\s*\n\s+json\.dump\([^,]+,\s*f[^)]*\)'
        write_replacement = r'\1safe_write_coordination_data(\2, \1)'
        
        # 这个替换比较复杂，暂时跳过自动替换，提供手动建议
        
        # 写入修复后的文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 修复完成!")
        print(f"📁 原文件备份: {backup_path}")
        print(f"📝 修复后文件: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        # 恢复备份
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, filepath)
            print(f"🔄 已从备份恢复原文件")
        return False

def create_manual_fix_guide():
    """创建手动修复指南"""
    guide = """
## 多进程JSON错误手动修复指南

### 问题描述
多进程环境下并发读写JSON文件时，可能出现JSON解析错误，通常是由于文件正在被写入时被其他进程读取导致的。

### 修复步骤

#### 1. 添加导入语句（在文件开头）
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from multiprocess_utils import safe_read_coordination_data, safe_write_coordination_data
```

#### 2. 替换读取代码
将以下代码：
```python
try:
    with open(unprocessed_data_file, 'r') as f:
        coordination_data = json.load(f)
```

替换为：
```python
try:
    coordination_data = safe_read_coordination_data(unprocessed_data_file)
    if not coordination_data:
        coordination_data = {"unprocessed_data": []}
```

#### 3. 添加数据验证
在处理数据时添加验证：
```python
unprocessed_data = []
for item_list in coordination_data.get("unprocessed_data", []):
    if isinstance(item_list, (list, tuple)):
        unprocessed_data.append(tuple(item_list))
    else:
        logging.warning(f"Skipping invalid item: {item_list}")
```

#### 4. 替换写入代码（如果存在）
将：
```python
with open(filepath, 'w') as f:
    json.dump(data, f)
```

替换为：
```python
safe_write_coordination_data(data, filepath)
```

### 修复原理
- **文件锁**: 确保读写操作的原子性
- **重试机制**: 遇到错误时自动重试
- **原子写入**: 使用临时文件避免写入过程中的数据损坏
- **数据验证**: 确保读取的数据格式正确

### 测试修复效果
运行修复后的代码，观察是否还会出现JSON解析错误。
"""
    
    with open("multiprocess_json_fix_guide.md", 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("📖 已创建手动修复指南: multiprocess_json_fix_guide.md")

def main():
    """主函数"""
    print("🚀 多进程JSON错误修复工具")
    print("=" * 50)
    
    # 检查是否存在multiprocess_utils.py
    utils_file = "utils/multiprocess_utils.py"
    if not os.path.exists(utils_file):
        print(f"❌ 依赖文件不存在: {utils_file}")
        print("请确保已经创建了multiprocess_utils.py文件")
        return False
    
    # 尝试修复pred_rag.py
    pred_rag_file = "benchmark/pred_rag.py"
    if os.path.exists(pred_rag_file):
        success = apply_fix_to_pred_rag(pred_rag_file)
        if success:
            print("\n🎉 自动修复成功!")
        else:
            print("\n⚠️  自动修复失败，请参考手动修复指南")
    else:
        print(f"⚠️  未找到 {pred_rag_file} 文件")
    
    # 创建手动修复指南
    create_manual_fix_guide()
    
    print("\n✨ 修复工具运行完成!")
    print("如果仍然遇到问题，请检查:")
    print("1. multiprocess_utils.py是否正确创建")
    print("2. 文件路径是否正确")
    print("3. 是否有足够的文件权限")

if __name__ == "__main__":
    main() 