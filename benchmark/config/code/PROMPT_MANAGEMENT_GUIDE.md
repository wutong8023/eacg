# Prompt管理系统使用指南

## 📋 概述

新的Prompt管理系统提供了一个结构化、可扩展的方式来管理VersiBCB项目中的所有prompt模板。该系统解决了原有系统中的以下问题：

- ❌ 命名混乱和版本号不一致
- ❌ 缺乏分类和组织
- ❌ 重复代码和冗余
- ❌ 缺乏文档和注释
- ❌ 难以维护和扩展

## 🏗️ 系统架构

### 1. 分类体系

```
├── 代码生成任务 (CODE_GENERATION)
│   └── VACECodeGeneration
│       ├── V1_BASIC - 基础版本
│       ├── V2_ENHANCED - 增强版本
│       ├── V3_ADVANCED - 高级版本
│       └── V4_PROFESSIONAL - 专业版本
│
├── 代码审查任务 (CODE_REVIEW)
│   ├── VACECodeReview
│   │   └── V1_BASIC - 基础审查
│   └── VSCCCodeReview
│       ├── V1_COMPREHENSIVE - 综合审查
│       ├── V2_ERROR_FIX_ONLY - 纯错误修复
│       ├── V3_WITH_FINAL_CODE - 带最终代码
│       └── V4_FROM_SCRATCH - 从零开始
│
├── 错误修复任务 (ERROR_FIX)
│   ├── VACEErrorFix
│   │   ├── V1_BASIC - 基础修复
│   │   └── V2_WITH_RETRIEVAL - 带检索修复
│   └── GeneralErrorFix
│       └── V1_PYTHON_ERROR - Python错误修复
│
└── 代码重构任务 (CODE_REFACTOR)
    └── (待扩展)
```

### 2. 命名规范

```
{TASK_TYPE}_{VERSION}_{FUNCTIONALITY}

示例：
- VACE_CODE_GENERATION
- VSCC_CODE_REVIEW
- VACE_ERROR_FIX
```

## 🚀 使用方法

### 1. 基本使用

```python
from benchmark.config.code.dataset2prompt_refactored import prompt_manager

# 获取VACE代码生成的高级版本
advanced_prompt = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V3_ADVANCED')

# 获取VSCC代码审查的comprehensive版本
comprehensive_review = prompt_manager.get_prompt('VSCC_CODE_REVIEW', 'V1_COMPREHENSIVE')

# 获取VACE错误修复的基础版本
error_fix_prompt = prompt_manager.get_prompt('VACE_ERROR_FIX', 'V1_BASIC')
```

### 2. 探索功能

```python
# 列出所有任务类型
task_types = prompt_manager.list_task_types()
print(task_types)
# ['VACE_CODE_GENERATION', 'VACE_CODE_REVIEW', 'VACE_ERROR_FIX', 'VSCC_CODE_REVIEW', 'GENERAL_ERROR_FIX']

# 列出特定任务类型的所有版本
versions = prompt_manager.list_versions('VACE_CODE_GENERATION')
print(versions)
# ['V1_BASIC', 'V2_ENHANCED', 'V3_ADVANCED', 'V4_PROFESSIONAL']

# 获取所有任务的信息
task_info = prompt_manager.get_task_info()
```

### 3. 向后兼容性

```python
# 原有的变量名仍然可以使用
from benchmark.config.code.dataset2prompt_refactored import versiBCB_vace_prompt_override

# 或者使用原有的映射
from benchmark.config.code.dataset2prompt_refactored import dataset2prompt
```

## 🔧 Prompt版本说明

### VACE代码生成系列

| 版本 | 特点 | 使用场景 |
|------|------|----------|
| V1_BASIC | 基础代码重构功能 | 简单的代码转换任务 |
| V2_ENHANCED | 添加功能对齐要求 | 需要确保功能一致性的任务 |
| V3_ADVANCED | 详细约束和指导 | 复杂的重构任务，需要精细控制 |
| V4_PROFESSIONAL | 支持API文档 | 专业级任务，利用API文档 |

### VSCC代码审查系列

| 版本 | 特点 | 使用场景 |
|------|------|----------|
| V1_COMPREHENSIVE | 全面的错误修复 | 需要完整代码审查的任务 |
| V2_ERROR_FIX_ONLY | 仅错误修复 | 专注于错误修复的任务 |
| V3_WITH_FINAL_CODE | 带最终代码输出 | 需要最终代码的任务 |
| V4_FROM_SCRATCH | 从零开始创建 | 需要重新创建代码的任务 |

### VACE错误修复系列

| 版本 | 特点 | 使用场景 |
|------|------|----------|
| V1_BASIC | 基础错误修复 | 简单的错误修复任务 |
| V2_WITH_RETRIEVAL | 带检索信息的修复 | 需要参考API信息的修复 |

## 🔄 迁移指南

### 从旧系统迁移到新系统

#### 步骤1：更新导入
```python
# 旧方式
from benchmark.config.code.dataset2prompt import versiBCB_vace_prompt_override

# 新方式
from benchmark.config.code.dataset2prompt_refactored import prompt_manager
```

#### 步骤2：更新使用方式
```python
# 旧方式
prompt = versiBCB_vace_prompt_override

# 新方式
prompt = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V1_BASIC')
```

#### 步骤3：版本映射对照表

| 旧变量名 | 新任务类型 | 新版本 |
|----------|-----------|--------|
| `versiBCB_vace_prompt_override` | `VACE_CODE_GENERATION` | `V1_BASIC` |
| `versiBCB_vace_prompt_override_v1` | `VACE_CODE_GENERATION` | `V2_ENHANCED` |
| `versiBCB_vace_prompt_override_v2` | `VACE_CODE_GENERATION` | `V3_ADVANCED` |
| `versiBCB_vace_prompt_override_v3` | `VACE_CODE_GENERATION` | `V4_PROFESSIONAL` |
| `VersiBCB_VACE_RAG_complete_withTargetCode_v1_review` | `VACE_CODE_REVIEW` | `V1_BASIC` |
| `VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review` | `VSCC_CODE_REVIEW` | `V1_COMPREHENSIVE` |
| `VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly` | `VSCC_CODE_REVIEW` | `V2_ERROR_FIX_ONLY` |
| `VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc` | `VSCC_CODE_REVIEW` | `V3_WITH_FINAL_CODE` |
| `VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode` | `VSCC_CODE_REVIEW` | `V4_FROM_SCRATCH` |
| `versiBCB_VACE_Prompt_errorfix` | `VACE_ERROR_FIX` | `V1_BASIC` |
| `versiBCB_VACE_Prompt_errorfix_retrieve` | `VACE_ERROR_FIX` | `V2_WITH_RETRIEVAL` |
| `VACE_Prompt_withPyError` | `GENERAL_ERROR_FIX` | `V1_PYTHON_ERROR` |

## 📝 添加新Prompt

### 步骤1：选择合适的类
```python
# 添加到现有类
class VACECodeGeneration:
    # 添加新版本
    V5_EXPERIMENTAL = '''
    新的prompt内容...
    '''
```

### 步骤2：更新PromptManager
```python
class PromptManager:
    def __init__(self):
        self.prompts = {
            'VACE_CODE_GENERATION': {
                # 现有版本...
                'V5_EXPERIMENTAL': VACECodeGeneration.V5_EXPERIMENTAL,
            },
            # 其他任务...
        }
```

### 步骤3：添加向后兼容性映射（如需要）
```python
# 如果需要向后兼容
new_prompt_name = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V5_EXPERIMENTAL')
```

## 🎯 最佳实践

### 1. 版本选择
- 🔰 **新手**：使用V1_BASIC版本
- 🏗️ **开发**：使用V2_ENHANCED或V3_ADVANCED版本
- 🚀 **生产**：使用V4_PROFESSIONAL版本

### 2. 性能优化
- 缓存常用的prompt以避免重复获取
- 使用适当的版本，不要过度复杂化

### 3. 错误处理
```python
try:
    prompt = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V3_ADVANCED')
except ValueError as e:
    print(f"Prompt获取失败: {e}")
    # 使用默认版本
    prompt = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V1_BASIC')
```

## 🔍 调试和测试

### 运行演示代码
```bash
python benchmark/config/code/dataset2prompt_refactored.py
```

### 验证向后兼容性
```python
# 确保所有原有变量仍然可用
from benchmark.config.code.dataset2prompt_refactored import dataset2prompt
assert len(dataset2prompt) > 0
```

## 🤝 贡献指南

### 添加新的任务类型
1. 创建新的类（如`NewTaskType`）
2. 定义版本常量
3. 更新`PromptManager`
4. 添加测试用例
5. 更新文档

### 添加新版本
1. 在现有类中添加新版本
2. 更新`PromptManager`映射
3. 更新版本说明文档
4. 添加使用示例

## 📚 参考资料

- [原始dataset2prompt.py](./dataset2prompt.py)
- [重构后的系统](./dataset2prompt_refactored.py)
- [多进程安全修复文档](../../MULTIPROCESS_JSON_FIX.md)

---

📞 **支持**：如有问题或建议，请提交Issue或联系项目维护者。 