#!/usr/bin/env python3
"""
Prompt管理系统测试脚本
=====================

该脚本用于验证重构后的prompt管理系统的正确性和向后兼容性。
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_prompt_manager():
    """测试PromptManager的基本功能"""
    print("🔍 Testing PromptManager basic functionality...")
    
    try:
        from benchmark.config.code.dataset2prompt_refactored import prompt_manager
        
        # 测试1: 获取任务类型
        task_types = prompt_manager.list_task_types()
        expected_types = ['VACE_CODE_GENERATION', 'VACE_CODE_REVIEW', 'VACE_ERROR_FIX', 'VSCC_CODE_REVIEW', 'GENERAL_ERROR_FIX']
        
        print(f"   ✓ Task types: {task_types}")
        assert all(t in task_types for t in expected_types), f"Missing task types: {set(expected_types) - set(task_types)}"
        
        # 测试2: 获取版本列表
        vace_versions = prompt_manager.list_versions('VACE_CODE_GENERATION')
        expected_versions = ['V1_BASIC', 'V2_ENHANCED', 'V3_ADVANCED', 'V4_PROFESSIONAL']
        
        print(f"   ✓ VACE versions: {vace_versions}")
        assert all(v in vace_versions for v in expected_versions), f"Missing versions: {set(expected_versions) - set(vace_versions)}"
        
        # 测试3: 获取prompt内容
        prompt = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V1_BASIC')
        assert isinstance(prompt, str) and len(prompt) > 100, "Prompt should be a non-empty string"
        print(f"   ✓ Prompt length: {len(prompt)} chars")
        
        # 测试4: 获取任务信息
        task_info = prompt_manager.get_task_info()
        assert 'VACE_CODE_GENERATION' in task_info
        assert 'versions' in task_info['VACE_CODE_GENERATION']
        assert 'total_versions' in task_info['VACE_CODE_GENERATION']
        print(f"   ✓ Task info: {len(task_info)} task types")
        
        # 测试5: 错误处理
        try:
            prompt_manager.get_prompt('NONEXISTENT_TASK', 'V1_BASIC')
            assert False, "Should raise ValueError for unknown task type"
        except ValueError:
            print("   ✓ Error handling for unknown task type")
        
        try:
            prompt_manager.get_prompt('VACE_CODE_GENERATION', 'NONEXISTENT_VERSION')
            assert False, "Should raise ValueError for unknown version"
        except ValueError:
            print("   ✓ Error handling for unknown version")
        
        print("✅ PromptManager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ PromptManager test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        # 测试原有变量名是否仍然可用
        from benchmark.config.code.dataset2prompt_refactored import (
            versiBCB_vace_prompt_override,
            versiBCB_vace_prompt_override_v1,
            versiBCB_vace_prompt_override_v2,
            versiBCB_vace_prompt_override_v3,
            VersiBCB_VACE_RAG_complete_withTargetCode_v1_review,
            VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review,
            VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly,
            VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc,
            VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode,
            versiBCB_VACE_Prompt_errorfix,
            versiBCB_VACE_Prompt_errorfix_retrieve,
            VACE_Prompt_withPyError,
            dataset2prompt
        )
        
        # 验证变量类型和内容
        variables_to_test = [
            ('versiBCB_vace_prompt_override', versiBCB_vace_prompt_override),
            ('versiBCB_vace_prompt_override_v1', versiBCB_vace_prompt_override_v1),
            ('versiBCB_vace_prompt_override_v2', versiBCB_vace_prompt_override_v2),
            ('versiBCB_vace_prompt_override_v3', versiBCB_vace_prompt_override_v3),
            ('VersiBCB_VACE_RAG_complete_withTargetCode_v1_review', VersiBCB_VACE_RAG_complete_withTargetCode_v1_review),
            ('VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review', VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review),
            ('VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly', VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly),
            ('VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc', VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc),
            ('VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode', VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode),
            ('versiBCB_VACE_Prompt_errorfix', versiBCB_VACE_Prompt_errorfix),
            ('versiBCB_VACE_Prompt_errorfix_retrieve', versiBCB_VACE_Prompt_errorfix_retrieve),
            ('VACE_Prompt_withPyError', VACE_Prompt_withPyError),
        ]
        
        for var_name, var_value in variables_to_test:
            assert isinstance(var_value, str), f"{var_name} should be a string"
            assert len(var_value) > 50, f"{var_name} should be a non-empty prompt"
            print(f"   ✓ {var_name}: {len(var_value)} chars")
        
        # 验证dataset2prompt映射
        assert isinstance(dataset2prompt, dict), "dataset2prompt should be a dictionary"
        assert len(dataset2prompt) > 0, "dataset2prompt should not be empty"
        print(f"   ✓ dataset2prompt: {len(dataset2prompt)} entries")
        
        # 验证映射中的所有值都是字符串
        for key, value in dataset2prompt.items():
            assert isinstance(value, str), f"dataset2prompt[{key}] should be a string"
            assert len(value) > 50, f"dataset2prompt[{key}] should be a non-empty prompt"
        
        print("✅ Backward compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_prompt_content_integrity():
    """测试prompt内容完整性"""
    print("\n🔍 Testing prompt content integrity...")
    
    try:
        from benchmark.config.code.dataset2prompt_refactored import prompt_manager
        
        # 测试所有prompt都包含必要的占位符
        test_cases = [
            ('VACE_CODE_GENERATION', 'V1_BASIC', ['{context}', '{description}', '{origin_dependency}', '{origin_code}', '{target_dependency}']),
            ('VACE_CODE_GENERATION', 'V3_ADVANCED', ['{context}', '{description}', '{origin_dependency}', '{origin_code}', '{target_dependency}']),
            ('VSCC_CODE_REVIEW', 'V1_COMPREHENSIVE', ['{description}', '{dependency}', '{generated_target_code}', '{error_info}', '{context}']),
            ('VACE_ERROR_FIX', 'V1_BASIC', ['{description}', '{target_dependency}', '{generated_target_code}', '{error_info}']),
        ]
        
        for task_type, version, expected_placeholders in test_cases:
            prompt = prompt_manager.get_prompt(task_type, version)
            
            for placeholder in expected_placeholders:
                assert placeholder in prompt, f"Missing placeholder '{placeholder}' in {task_type} {version}"
            
            print(f"   ✓ {task_type} {version}: All placeholders present")
        
        # 测试prompt结构
        vace_basic = prompt_manager.get_prompt('VACE_CODE_GENERATION', 'V1_BASIC')
        assert 'You are now a professional Python programming engineer' in vace_basic
        assert 'Context from target dependency' in vace_basic
        assert 'Refactored new code' in vace_basic
        print("   ✓ Prompt structure validation passed")
        
        print("✅ Prompt content integrity tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Prompt content integrity test failed: {e}")
        traceback.print_exc()
        return False

def test_new_vs_old_system():
    """比较新旧系统的输出"""
    print("\n⚖️  Comparing new vs old system...")
    
    try:
        # 尝试导入旧系统（如果存在）
        old_system_available = True
        try:
            from benchmark.config.code.dataset2prompt import versiBCB_vace_prompt_override as old_prompt
        except ImportError:
            old_system_available = False
            print("   ℹ️  Old system not available for comparison")
        
        from benchmark.config.code.dataset2prompt_refactored import versiBCB_vace_prompt_override as new_prompt
        
        if old_system_available:
            # 比较内容
            assert isinstance(old_prompt, str), "Old prompt should be string"
            assert isinstance(new_prompt, str), "New prompt should be string"
            
            # 内容应该相似（允许格式化差异）
            old_words = set(old_prompt.lower().split())
            new_words = set(new_prompt.lower().split())
            common_words = old_words & new_words
            similarity = len(common_words) / max(len(old_words), len(new_words))
            
            print(f"   ✓ Content similarity: {similarity:.2%}")
            assert similarity > 0.8, f"Content similarity too low: {similarity:.2%}"
        else:
            print("   ✓ New system prompt loaded successfully")
        
        print("✅ New vs old system comparison passed!")
        return True
        
    except Exception as e:
        print(f"❌ New vs old system comparison failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 Starting Prompt Management System Tests")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("PromptManager Basic Functionality", test_prompt_manager),
        ("Backward Compatibility", test_backward_compatibility),
        ("Prompt Content Integrity", test_prompt_content_integrity),
        ("New vs Old System", test_new_vs_old_system),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        test_results.append((test_name, result))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary:")
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Overall: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The prompt management system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the system.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 