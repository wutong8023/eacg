"""
VersiBCB代码生成和修复任务的Prompt管理系统
=========================================

该模块包含用于不同代码生成和修复任务的prompt模板。
所有prompt都按照功能分类，并使用统一的命名规范。

命名规范：
- {TASK}_{VERSION}_{FUNCTIONALITY}
- 例如：VACE_V1_CODE_GENERATION, VSCC_V2_ERROR_FIX

功能分类：
1. CODE_GENERATION - 代码生成任务
2. CODE_REVIEW - 代码审查任务  
3. ERROR_FIX - 错误修复任务
4. CODE_REFACTOR - 代码重构任务
"""

# =============================================================================
# 1. 代码生成任务 (CODE_GENERATION)
# =============================================================================

class VACECodeGeneration:
    """VACE任务的代码生成prompt集合"""
    
    # 基础版本 - 简单的代码重构
    V1_BASIC = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.

I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version.

### Context from target dependency
{context}

### Functionality description of the code
{description}

### Dependency and old version
{origin_dependency}

### Old version code
{origin_code}

### Dependency and new version
{target_dependency}

### Instructions
The provided context may contain both relevant and irrelevant information for achieving the code functionality. Please carefully evaluate whether to use these references, always prioritizing fulfillment of the code functionality requirement.

Please only return the refactored code and enclose it with `<start>` and `<end>`.

### Refactored new code
'''

    # 增强版本 - 添加了功能对齐要求
    V2_ENHANCED = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.

I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version.

### Context from target dependency
{context}

### Functionality description of the code
{description}

### Dependency and old version
{origin_dependency}

### Old version code
{origin_code}

### Dependency and new version
{target_dependency}

### Instructions
1. The provided context may contain both relevant and irrelevant information for achieving the code functionality. Please carefully evaluate whether to use these references, always prioritizing fulfillment of the code functionality requirement.
2. Please make sure the refactored code can meet the functionality requirement in target dependency, which aligns with the functionality achieved by the old code in old version.

Please only return the refactored code and enclose it with `<start>` and `<end>`.

### Refactored new code
'''

    # 高级版本 - 详细的约束和指导
    V3_ADVANCED = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.

I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version.

### Context from target dependency
{context}

### Functionality description of the code
{description}

### Dependency and old version
{origin_dependency}

### Old version code
{origin_code}

### Dependency and new version
{target_dependency}

### Important Guidelines
1. **Context Evaluation**: The provided context may contain both relevant and irrelevant information for achieving the code functionality, or all info could be completely irrelevant to the code functionality. Please carefully consider whether to use these content, always prioritizing fulfillment of the code functionality requirement.

2. **Functionality Alignment**: Please make sure the refactored code can meet the functionality requirement in target dependency, which aligns with the functionality achieved by the old code in old version.

3. **Minimal Changes**: Try to make the least change to the old code to achieve the functionality requirement in target dependency to avoid potential mistakes. Try to avoid unnecessary imports if it is not needed for achieving target functionality.

4. **Import Considerations**: Some paths of retrieved content cannot be imported because they could be methods, properties of class.

5. **Function Naming**: Use 'task_func' as the function name of the refactored code, do not change the function name and make sure all necessary imports are included to avoid import errors.

6. **Output Format**: Please only return the refactored code and enclose it with `<start>` and `<end>`.

### Refactored new code
'''

    # 专业版本 - 添加API文档支持
    V4_PROFESSIONAL = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.

I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version.

### Context from target dependency
{context}

### Functionality description of the code
{description}

### Dependency and old version
{origin_dependency}

### Old version code
{origin_code}

### Dependency and new version
{target_dependency}

### Important Guidelines
1. **API Documentation**: The provided context now contains docstrings which indicate the usage of some available APIs from target dependency. The provided context may contain both relevant and irrelevant information for achieving the code functionality, or all info could be completely irrelevant to the code functionality.

2. **Functionality Preservation**: Please make sure the refactored code can meet the functionality requirement in target dependency, which aligns with the functionality achieved by the old code in old version.

3. **Minimal Changes**: Try to make the least change to the old code to achieve the functionality requirement in target dependency to avoid potential mistakes. Try to avoid unnecessary imports if it is not needed for achieving target functionality.

4. **Import Restrictions**: Some paths of retrieved content cannot be imported because they could be methods, properties of class.

5. **Function Standards**: Use 'task_func' as the function name of the refactored code, do not change the function name and make sure all necessary imports are included to avoid import errors.

6. **Output Format**: Please only return the refactored code and enclose it with `<start>` and `<end>`.

### Refactored new code
'''


# =============================================================================
# 2. 代码审查任务 (CODE_REVIEW)
# =============================================================================

class VACECodeReview:
    """VACE任务的代码审查prompt集合"""
    
    V1_BASIC = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model.

I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code.

### Related content of target dependency
{context}

### Functionality description of the code
{description}

### Dependency and target new version
{target_dependency}

### Generated code
{generated_target_code}

Please only return the revised code and enclose it with `<start>` and `<end>`.

### New version code after revision
(If no revision needed, only return the wrapper, which should be '<start><end>')
'''


class VSCCCodeReview:
    """VSCC任务的代码审查prompt集合"""
    
    V1_COMPREHENSIVE = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info.

### Functionality description of the code
{description}

### Dependency and target new version
{dependency}

### To-be-fix code
{generated_target_code}

### Code error info detected by static analysis in target_dependency
{error_info}

### Retrieved context that may be related to error
{context}

Please analyse the error, then return the revised code and enclose the revised code with `<start>` and `<end>`. Mark out the revised part using comment following python comment style.

### Important Notes
1. **Import Management**: Keep an eye on the imports, try to avoid errors due to no-imported package usage in the code (for example, if the to-be-fix code uses 'datetime' but does not have 'datetime', you should add 'import datetime' to the code).

2. **Attribute Errors**: Focus on error info that assert attribute missing (for example, AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'), try to use the equivalent found in retrieved info to fix the error if there is any (for example, the error info is AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names', you should use the equivalent 'get_feature_names_out' found in retrieved info to fix the error).

3. **Python Version Compatibility**: Python 3.5 has a different str format behavior. For example, f"x is {{x}}" is not valid in python 3.5, but valid in python 3.6. So, if target dependency is python 3.5, you should use the old str format behavior (which means you should use 'x is {{}}'.format(x) instead of f"x is {{x}}" if python 3.5 is in the target dependency).

### Revised code which avoids the error
'''

    V2_ERROR_FIX_ONLY = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info while keeping its original functionality.

### Dependency and target new version
{dependency}

### To-be-fix code
{generated_target_code}

### Code error info detected by static analysis in target_dependency
{error_info}

### Retrieved context that may be related to error
{context}

Please analyse the error, give possible fix for each error, in the format of old_codeline to new_codeline, which is wrapped in <errorfix></errorfix>.

Below is an example of the format:
**Error Info:**
Line 12: return f'Process found. Restarting {{process_name}}.'
Error: Format strings are only supported in Python 3.6 and greater
In python3.5, as an example, should use "{{}} {{}}".format(x,y) to replace f"{{x}} {{y}}"

<errorfix>
old line: 
Line 12: return f'Process found. Restarting {{process_name}}.'
new line:
Line 12: return 'Process found. Restarting {{}}.'.format(process_name)
</errorfix>

### Error fixes
'''

    V3_WITH_FINAL_CODE = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info while keeping its original functionality.

### Dependency and target new version
{dependency}

### To-be-fix code
{generated_target_code}

### Code error info detected by static analysis in target_dependency
{error_info}

### Retrieved context that may be related to error
{context}

Please analyse the error, give possible fix for each error, in the format of old_codeline to new_codeline, which is wrapped in <errorfix></errorfix>. Then, return the final code in <finalcode></finalcode> wrapper.

Below is an example of the format:
**Error Info:**
Line 12: return f'Process found. Restarting {{process_name}}.'
Error: Format strings are only supported in Python 3.6 and greater
In python3.5, as an example, should use "{{}} {{}}".format(x,y) to replace f"{{x}} {{y}}"

<errorfix>
old line: 
Line 12: return f'Process found. Restarting {{process_name}}.'
new line:
Line 12: return 'Process found. Restarting {{}}.'.format(process_name)
</errorfix>

Now, generate the error fixes and return the final code in <finalcode></finalcode> wrapper.
'''

    V4_FROM_SCRATCH = '''
You are now a professional Python programming engineer. Your task is to create code that caters to functionality requirement in target dependency.

### Functionality requirement of the code
{description}

### Dependency and target new version
{dependency}

### Code error info detected by static analysis in target_dependency
{error_info}

### Retrieved context that may be related to error
{context}

Please enclose the code with `<start>` and `<end>`.

### Important Notes
1. **Import Management**: Keep an eye on the imports, try to avoid errors due to no-imported package usage in the code (for example, if the code uses 'datetime' but does not have 'datetime', you should add 'import datetime' to the code).

2. **Attribute Errors**: Focus on error info that assert attribute missing (for example, AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'), try to use the equivalent found in retrieved info to fix the error if there is any (for example, the error info is AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names', you should use the equivalent 'get_feature_names_out' found in retrieved info to fix the error).

3. **Python Version Compatibility**: Python 3.5 has a different str format behavior. For example, f"x is {{x}}" is not valid in python 3.5, but valid in python 3.6. So, if target dependency is python 3.5, you should use the old str format behavior (which means you should use 'x is {{}}'.format(x) instead of f"x is {{x}}" if python 3.5 is in the target dependency).

### Code that caters to functionality requirement in target dependency
'''


# =============================================================================
# 3. 错误修复任务 (ERROR_FIX)
# =============================================================================

class VACEErrorFix:
    """VACE任务的错误修复prompt集合"""
    
    V1_BASIC = '''
You are an expert Python programmer. Please edit the target code to fix the errors while preserving the original functionality.

### Functionality description of the code
{description}

### Dependency version the code needs to run on
{target_dependency}

### Generated code
{generated_target_code}

### Detected errors in the generated code
(Note: Some errors detected may be just warnings, which do not affect the code.)
{error_info}

Now, fix the error given the error info and the dependency version.

### Fixed target code
(wrapped by '```python' and '```')
'''

    V2_WITH_RETRIEVAL = '''
You are an expert Python programmer. Please edit the target code to fix the errors while preserving the original functionality.

### Functionality description of the code
{description}

### Dependency version the code needs to run on
{target_dependency}

### Generated code
{generated_target_code}

### Detected errors in the generated code
(Note: Some errors detected may be just warnings, which do not affect the code.)
{error_info}

### Similar API paths found in target dependency
(Could be related or unrelated since only sequence-based similarity is used)
{retrieved_info}

---

Now, fix the error given the error info and the dependency version.

### Fixed target code
(wrapped by '```python' and '```')
'''


class GeneralErrorFix:
    """通用错误修复prompt"""
    
    V1_PYTHON_ERROR = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.

I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version.

### Related content of target dependency
{context}

### Functionality description of the code
{description}

### Dependency and old version
{origin_dependency}

### Old version code
{origin_code}

### Dependency and new version
{target_dependency}

Please only return the refactored code and enclose it with `<start>` and `<end>`.

### Refactored new code
'''


# =============================================================================
# 4. Prompt映射和管理系统
# =============================================================================

class PromptManager:
    """Prompt管理系统，提供统一的接口来访问不同类型的prompt"""
    
    def __init__(self):
        self.prompts = {
            # VACE任务
            'VACE_CODE_GENERATION': {
                'V1_BASIC': VACECodeGeneration.V1_BASIC,
                'V2_ENHANCED': VACECodeGeneration.V2_ENHANCED,
                'V3_ADVANCED': VACECodeGeneration.V3_ADVANCED,
                'V4_PROFESSIONAL': VACECodeGeneration.V4_PROFESSIONAL,
            },
            'VACE_CODE_REVIEW': {
                'V1_BASIC': VACECodeReview.V1_BASIC,
            },
            'VACE_ERROR_FIX': {
                'V1_BASIC': VACEErrorFix.V1_BASIC,
                'V2_WITH_RETRIEVAL': VACEErrorFix.V2_WITH_RETRIEVAL,
            },
            
            # VSCC任务
            'VSCC_CODE_REVIEW': {
                'V1_COMPREHENSIVE': VSCCCodeReview.V1_COMPREHENSIVE,
                'V2_ERROR_FIX_ONLY': VSCCCodeReview.V2_ERROR_FIX_ONLY,
                'V3_WITH_FINAL_CODE': VSCCCodeReview.V3_WITH_FINAL_CODE,
                'V4_FROM_SCRATCH': VSCCCodeReview.V4_FROM_SCRATCH,
            },
            
            # 通用任务
            'GENERAL_ERROR_FIX': {
                'V1_PYTHON_ERROR': GeneralErrorFix.V1_PYTHON_ERROR,
            }
        }
    
    def get_prompt(self, task_type, version='V1_BASIC'):
        """
        获取指定任务类型和版本的prompt
        
        Args:
            task_type (str): 任务类型，如 'VACE_CODE_GENERATION'
            version (str): 版本号，如 'V1_BASIC'
            
        Returns:
            str: 对应的prompt模板
        """
        if task_type not in self.prompts:
            raise ValueError(f"Unknown task type: {task_type}")
        
        if version not in self.prompts[task_type]:
            raise ValueError(f"Unknown version '{version}' for task type '{task_type}'")
        
        return self.prompts[task_type][version]
    
    def list_task_types(self):
        """列出所有可用的任务类型"""
        return list(self.prompts.keys())
    
    def list_versions(self, task_type):
        """列出指定任务类型的所有版本"""
        if task_type not in self.prompts:
            raise ValueError(f"Unknown task type: {task_type}")
        return list(self.prompts[task_type].keys())
    
    def get_task_info(self):
        """获取所有任务的信息"""
        info = {}
        for task_type in self.prompts:
            info[task_type] = {
                'versions': list(self.prompts[task_type].keys()),
                'total_versions': len(self.prompts[task_type])
            }
        return info


# =============================================================================
# 5. 向后兼容性映射
# =============================================================================

# 创建prompt管理器实例
prompt_manager = PromptManager()

# 向后兼容性映射 - 保持原有的变量名可以正常使用
versiBCB_vace_prompt_override = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V1_BASIC')
versiBCB_vace_prompt_override_v1 = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V2_ENHANCED')
versiBCB_vace_prompt_override_v2 = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V3_ADVANCED')
versiBCB_vace_prompt_override_v3 = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V4_PROFESSIONAL')

VersiBCB_VACE_RAG_complete_withTargetCode_v1_review = prompt_manager.get_prompt('VACE_CODE_REVIEW', 'V1_BASIC')

VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review = prompt_manager.get_prompt('VSCC_CODE_REVIEW', 'V1_COMPREHENSIVE')
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly = prompt_manager.get_prompt('VSCC_CODE_REVIEW', 'V2_ERROR_FIX_ONLY')
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc = prompt_manager.get_prompt('VSCC_CODE_REVIEW', 'V3_WITH_FINAL_CODE')
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode = prompt_manager.get_prompt('VSCC_CODE_REVIEW', 'V4_FROM_SCRATCH')

versiBCB_VACE_Prompt_errorfix = prompt_manager.get_prompt('VACE_ERROR_FIX', 'V1_BASIC')
versiBCB_VACE_Prompt_errorfix_retrieve = prompt_manager.get_prompt('VACE_ERROR_FIX', 'V2_WITH_RETRIEVAL')

VACE_Prompt_withPyError = prompt_manager.get_prompt('GENERAL_ERROR_FIX', 'V1_PYTHON_ERROR')

# 原有的dataset2prompt映射保持不变
dataset2prompt = {
    'versiBCB_vace_prompt_override': versiBCB_vace_prompt_override,
    'versiBCB_vace_prompt_override_v1': versiBCB_vace_prompt_override_v1,
    'versiBCB_vace_prompt_override_v2': versiBCB_vace_prompt_override_v2,
    'versiBCB_vace_prompt_override_v3': versiBCB_vace_prompt_override_v3,
    'VersiBCB_VACE_RAG_complete_withTargetCode_v1_review': VersiBCB_VACE_RAG_complete_withTargetCode_v1_review,
    'VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review': VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review,
    'VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly': VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly,
    'VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc': VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc,
    'VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode': VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode,
    'versiBCB_VACE_Prompt_errorfix': versiBCB_VACE_Prompt_errorfix,
    'versiBCB_VACE_Prompt_errorfix_retrieve': versiBCB_VACE_Prompt_errorfix_retrieve,
    'VACE_Prompt_withPyError': VACE_Prompt_withPyError,
}


# =============================================================================
# 6. 使用示例和文档
# =============================================================================

if __name__ == "__main__":
    # 使用示例
    print("=== Prompt Management System Demo ===")
    
    # 1. 列出所有任务类型
    print("\n1. Available Task Types:")
    for task_type in prompt_manager.list_task_types():
        print(f"   - {task_type}")
    
    # 2. 获取任务信息
    print("\n2. Task Information:")
    task_info = prompt_manager.get_task_info()
    for task_type, info in task_info.items():
        print(f"   {task_type}: {info['total_versions']} versions")
        for version in info['versions']:
            print(f"     - {version}")
    
    # 3. 使用新的管理系统
    print("\n3. Usage Examples:")
    
    # 获取VACE代码生成的高级版本
    advanced_prompt = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V3_ADVANCED')
    print("   Advanced VACE Code Generation prompt loaded successfully")
    
    # 获取VSCC代码审查的comprehensive版本
    comprehensive_review = prompt_manager.get_prompt('VSCC_CODE_REVIEW', 'V1_COMPREHENSIVE')
    print("   Comprehensive VSCC Code Review prompt loaded successfully")
    
    # 4. 向后兼容性测试
    print("\n4. Backward Compatibility Test:")
    print("   Original variable names still work:")
    print(f"   versiBCB_vace_prompt_override length: {len(versiBCB_vace_prompt_override)} chars")
    print(f"   dataset2prompt keys: {len(dataset2prompt)} prompts") 