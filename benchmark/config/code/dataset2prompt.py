import json

# 从JSON文件加载prompt
with open("benchmark/config/dataset2prompt.json", "r") as f:
    json_prompts = json.load(f)
versiBCB_vace_prompt_override = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. \n### some retrieved content from target dependency\n{context}\n### Functionality description of the code\n{description}\n### Dependency and old version\n{origin_dependency}\n### Old version code\n{origin_code}\n### Dependency and new version\n{target_dependency}\n\nNotably,the provided context may contain both relevant and irrelevant information for achieving the code functionality. Please carefully evaluate whether to use these references, always prioritizing fullfillment of the code functionality requirement.Please only return the refactored code and enclose it with `<start>` and `<end>`.\n\n### Refactored new code\n
'''
versiBCB_vace_prompt_override_v1 = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. \n### some retrieved content from target dependency\n{context}\n### Functionality description of the code\n{description}\n### Dependency and old version\n{origin_dependency}\n### Old version code\n{origin_code}\n### Dependency and new version\n{target_dependency}\n\nNotably,the provided context may contain both relevant and irrelevant information for achieving the code functionality. Please carefully evaluate whether to use these references, always prioritizing fullfillment of the code functionality requirement.Please make sure the refactored code can meet the functionality requirement in target dependency, which align with the functionality achieved by the old code in old version.Please only return the refactored code and enclose it with `<start>` and `<end>`.\n\n### Refactored new code\n
'''
versiBCB_vace_prompt_override_v2 = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. \n### some retrieved content from target dependency\n{context}\n### Functionality description of the code\n{description}\n### Dependency and old version\n{origin_dependency}\n### Old version code\n{origin_code}\n### Dependency and new version\n{target_dependency}\n\n### Important Notes\n1.the provided context may contain both relevant and irrelevant information for achieving the code functionality, and all info could be completely irrelevant to the code functionality. Please carefully consider whether to use these content, always prioritizing fullfillment of the code functionality requirement.2.Please make sure the refactored code can meet the functionality requirement in target dependency, which align with the functionality achieved by the old code in old version.3.try to make the least change to the old code to achieve the functionality requirement in target dependency to avoid potential mistakes. try to aviod unnecessary imports if it is not needed for achieving target functionality. 4.some path of retrieved content cannot be imported because they could be methods,properties of class.
5.use 'task_func' as the function name of the refactored code, do not change the function name and make sure all necessary imports are included to avoid import errors.6.Please only return the refactored code and enclose it with `<start>` and `<end>`.\n\n### Refactored new code\n
'''
versiBCB_vace_prompt_override_v3 = '''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. \n### some context from target dependency\n{context}\n### Functionality description of the code\n{description}\n### Dependency and old version\n{origin_dependency}\n### Old version code\n{origin_code}\n### Dependency and new version\n{target_dependency}\n\n### Important Notes\n
1.the provided context now contain docstring which indicate the usage of some available apis from target dependency. the provided context may contain both relevant and irrelevant information for achieving the code functionality, or all info could be completely irrelevant to the code functionality.
2.Please make sure the refactored code can meet the functionality requirement in target dependency, which align with the functionality achieved by the old code in old version.
3.try to make the least change to the old code to achieve the functionality requirement in target dependency to avoid potential mistakes. try to aviod unnecessary imports if it is not needed for achieving target functionality.
4.some path of retrieved content cannot be imported because they could be methods,properties of class.
5.use 'task_func' as the function name of the refactored code, do not change the function name and make sure all necessary imports are included to avoid import errors.
6.Please only return the refactored code and enclose it with `<start>` and `<end>`.\n\n### Refactored new code\n
'''
VersiBCB_VACE_RAG_complete_withTargetCode_v1_review_v0 = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model.I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code.
### related content of target dependency{context}\n### Functionality description of the code\n{description}\n\n### Dependency and target new version\n{target_dependency}\n### generated code\n{generated_target_code}\nPlease only return the revised code and enclose it with `<start>` and `<end>`.\n\n### new version code after revision(if no revision needed,only return the wrapper, which should be  '<start><end>')\n
'''
# 下部暂时移除了### Functionality description of the code\n{description}\n\n
# 移除了### related content of target dependency{context}\n
# VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_old = '''
# You are now a professional Python programming engineer. Your task is to review the target code generated by the model. Try fix the error according to error_info
# ### Description of target dependency\n{description}\n
# ### Dependency and target new version\n{dependency}\n### generated code\n{generated_target_code}### code error info detected by static analysis in target_dependency(some info could just be warning) \n{error_info}\nPlease only return the revised code and enclose it with `<start>` and `<end>`.\n\n### new version code after revision(if no revision needed,only return the wrapper, which should be  '<start><end>')\n
# '''
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info.
### functionality description of the code\n{description}\n
### Dependency and target new version\n{dependency}\n### to-be-fix code\n{generated_target_code}\n### code error info detected by static analysis in target_dependency\n{error_info}### retrieved_context that may be related to error\n {context}\n\nPlease analyse the error, then return the revised code and enclose the revised code with `<start>` and `<end>`.mark out the revised part using comment following python comment style\n\n 
### notes
1.keep an eye on the imports, try to avoid errors due to no-imported package usage in the code(for example, if the to-be-fix code use 'datetime' but does not have 'datetime', you should add 'import datetime' to the code)
2.focus on error info that assert attribute missing(for example, AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'\n), try to use the equalvalence found in retrieved info to fix the error if there is any(for example, the error info is AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names', you should use the equalvalence 'get_feature_names_out' found in retrieved info to fix the error).
3.python 3.5 has a different str format behavior. for example, f"x is {{x}}" is not valid in python 3.5, but valid in python 3.6. So, if target dependency is python 3.5, you should use the old str format behavior(which means you should use 'x is {{}}'.format(x) instead of f"x is {{x}}" if python 3.5 is in the target dependency)
### revised code which avoid the error\n
'''
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorfixonly = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info while keeping its original functionality.
### Dependency and target new version\n{dependency}\n### to-be-fix code\n{generated_target_code}\n### code error info detected by static analysis in target_dependency\n{error_info}### retrieved_context that may be related to error\n {context}\n\nPlease analyse the error,give possible fix for each error,in the format of old_codeline to new_codeline. which is wrapped in <errorfix></errorfix>.
below is an example of the format:
errorinfo:
Line 12:             return f'Process found. Restarting {{process_name}}.'\nError: Format strings are only supported in Python 3.6 and greater\nIn python3.5, as an example, should use \"{{}} {{}}\".format(x,y) to replace f\"{{x}} {{y}}\"
<errorfix>
old line: 
Line 12:             return f'Process found. Restarting {{process_name}}.'
new line:
Line 12:             return f'Process found. Restarting {{}}.'.format(process_name)
</errorfix>
### errorfixes
'''
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info while keeping its original functionality.
### Dependency and target new version\n{dependency}\n### to-be-fix code\n{generated_target_code}\n### code error info detected by static analysis in target_dependency\n{error_info}### retrieved_context that may be related to error\n {context}\n\nPlease analyse the error,give possible fix for each error,in the format of old_codeline to new_codeline. which is wrapped in <errorfix></errorfix>. Then, return the final code in <finalcode></finalcode> wrapper.
below is an example of the format:
errorinfo:
Line 12:             return f'Process found. Restarting {{process_name}}.'\nError: Format strings are only supported in Python 3.6 and greater\nIn python3.5, as an example, should use \"{{}} {{}}\".format(x,y) to replace f\"{{x}} {{y}}\"
<errorfixs>
<errorfix>
old line: 
Line 12:             return f'Process found. Restarting {{process_name}}.'
new line:
Line 12:             return f'Process found. Restarting {{}}.'.format(process_name)
</errorfix>
</errorfixs>
So, you should generate the errorfixes and return the final code in <finalcode></finalcode> wrapper,which should be like:
### errorfixes and final code
<errorfixs>
<errorfix>...</errorfix>
<errorfix>...</errorfix>
...
</errorfixs>
<finalcode>
...
</finalcode>
Notably, errorfixes may be not sufficient to cover all errors since errors by static analysis are only the first error detected. so in the finalcode , except from errorfixes, you should keep on fixing the code. Especially take care about the same type error that have been solved in errorfix.
now, generate the errorfixes and give the final code that fix the errors. 
### errorfixes and final code
'''
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_wdesc = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info while keeping its original functionality.
### Dependency and target new version\n{dependency}\n###functionality description of the code\n{description}\n### to-be-fix code\n{generated_target_code}\n### code error info detected by static analysis in target_dependency\n{error_info}### retrieved_context that may be related to error\n {context}\n\nPlease analyse the error,give possible fix for each error,in the format of old_codeline to new_codeline. which is wrapped in <errorfix></errorfix>. Then, return the final code in <finalcode></finalcode> wrapper.
below is an example of the format:
errorinfo:
Line 12:             return f'Process found. Restarting {{process_name}}.'\nError: Format strings are only supported in Python 3.6 and greater\nIn python3.5, as an example, should use \"{{}} {{}}\".format(x,y) to replace f\"{{x}} {{y}}\"
<errorfixs>
<errorfix>
old line: 
Line 12:             return f'Process found. Restarting {{process_name}}.'
new line:
Line 12:             return f'Process found. Restarting {{}}.'.format(process_name)
</errorfix>
</errorfixs>
So, you should generate the errorfixes and return the final code in <finalcode></finalcode> wrapper,which should be like:
### errorfixes and final code
<errorfixs>
<errorfix>...</errorfix>
<errorfix>...</errorfix>
...
</errorfixs>
<finalcode>
...
</finalcode>
Notably, errorfixes may be not sufficient to cover all errors since errors by static analysis are only the first error detected. so in the finalcode , except from errorfixes, you should keep on fixing the code. Especially take care about the same type error that have been solved in errorfix.
now, generate the errorfixes and give the final code that fix the errors. 
### errorfixes and final code
'''
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nodesc_errorinduced = '''
You are now a professional Python programming engineer. Your task is to review the target code generated by the model, fix the error and give the correct code that is complete and runnable in target dependency according to error_info and retrieved info while keeping its original functionality.
### Dependency and target new version\n{dependency}\n### to-be-fix code\n{generated_target_code}\n### code error info detected by static analysis in target_dependency\n{error_info}### retrieved_context that may be related to error\n {context}\n\nPlease analyse the error,induce same errors as the static analysed errors,give possible fix for each error,in the format of old_codeline to new_codeline. which is wrapped in <errorfix></errorfix>. Then, return the final code in <finalcode></finalcode> wrapper.
below is an example of the errorfix format:
errorinfo:
Line 12:             return f'Process found. Restarting {{process_name}}.'\nError: Format strings are only supported in Python 3.6 and greater\nIn python3.5, as an example, should use \"{{}} {{}}\".format(x,y) to replace f\"{{x}} {{y}}\"
<errorfixs>
<errorfix>
old line: 
Line 12:             return f'Process found. Restarting {{process_name}}.'
new line:
Line 12:             return f'Process found. Restarting {{}}.'.format(process_name)
</errorfix>
</errorfixs>
you should generate induced errors(errors that can be induced from errors that are detected by static analysis (follow same format of the errors in static analysed errors)) and  errorfixes(for induced errors and static analysed errors) and return the final code in <finalcode></finalcode> wrapper,which should be like:
### induced errors
<induced_errors>
<error>
...
</error>
<error>
...
</error>
</induced_errors>
### errorfixes and final code
<errorfixs>
<errorfix>...</errorfix>
<errorfix>...</errorfix>
...
</errorfixs>
<finalcode>
...
</finalcode>
Notably, errorfixes may be not sufficient to cover all errors since errors by static analysis are only the first error detected. so in the finalcode , except from errorfixes, you should keep on fixing the code. Especially take care about the same type error that have been solved in errorfix.
now, generate the errorfixes and give the final code that fix the errors. 
### induced errors
'''
VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_nocode = '''
You are now a professional Python programming engineer. Your task is to  create code that cater to functionality requirement in target dependency.
### functionality requirement of the code\n{description}\n
### Dependency and target new version\n{dependency}\n\n### code error info detected by static analysis in target_dependency\n{error_info}### retrieved_context that may be related to error\n {context}\n\nPlease enclose the code with `<start>` and `<end>`.
### notes
1.keep an eye on the imports, try to avoid errors due to no-imported package usage in the code(for example, if the code use 'datetime' but does not have 'datetime', you should add 'import datetime' to the code)
2.focus on error info that assert attribute missing(for example, AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'\n), try to use the equalvalence found in retrieved info to fix the error if there is any(for example, the error info is AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names', you should use the equalvalence 'get_feature_names_out' found in retrieved info to fix the error).
3.python 3.5 has a different str format behavior. for example, f"x is {{x}}" is not valid in python 3.5, but valid in python 3.6. So, if target dependency is python 3.5, you should use the old str format behavior(which means you should use 'x is {{}}'.format(x) instead of f"x is {{x}}" if python 3.5 is in the target dependency)
### code that cater to functionality requirement in target dependency\n
'''
versiBCB_VACE_Prompt_errorfix = '''
You are an expert Python programmer. Please edit the target code to fix the errors while reserving the original functionality.
### Functionality description of the code
{description}
### dependency version the code need to run on
{target_dependency}
### generated code
{generated_target_code}
### detected errors in the generated code(Notably, some error detected may be just warnings , which do not affect the code. )
{error_info}
Now, fix the error given the error info and the dependency version. 
### fixed target code(wrapped by '```python' and '```')

'''
versiBCB_VACE_Prompt_errorfix_retrieve = '''
You are an expert Python programmer. Please edit the target code to fix the errors while reserving the original functionality.
### Functionality description of the code
{description}
### dependency version the code need to run on
{target_dependency}
### generated code
{generated_target_code}
### detected errors in the generated code(Notably, some error detected may be just warnings , which do not affect the code. )
{error_info}
### similar API paths found in target dependency(could be related or unrelated since only sequence-based similarity is used)
{retrieved_info}
---
Now, fix the error given the error info and the dependency version. 
### fixed target code(wrapped by '```python' and '```')

'''
VACE_Prompt_withPyError='''
You are now a professional Python programming engineer. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. \n### related content of target dependency\n{context}\n### Functionality description of the code\n{description}\n### Dependency and old version\n{origin_dependency}\n### Old version code\n{origin_code}\n### Dependency and new version\n{target_dependency}\n\nPlease only return the refactored code and enclose it with `<start>` and `<end>`.\n\n### Refactored new code\n
'''
dataset2prompt = {
    "VersiCode": {
        "VSCC": json_prompts["Versicode_vscc"].replace("{context}", "{knowledge_doc}"),
        "VACE": json_prompts["Versicode_vace"].replace("{context}", "{knowledge_doc}")
    },
    "VersiBCB": {
        "VSCC_GEN": json_prompts["VersiBCB_vscc_GEN"].replace("{context}", "{knowledge_doc}"),
        "VACE_GEN": json_prompts["VersiBCB_vace_GEN"].replace("{context}", "{knowledge_doc}"),
        "VSCC": json_prompts["VersiBCB_vscc"].replace("{context}", "{knowledge_doc}"),
        "VACE": json_prompts["VersiBCB_vace"].replace("{context}", "{knowledge_doc}"),
        # "VACE": versiBCB_vace_prompt_override_v2.replace("{context}", "{knowledge_doc}"),
        "VSCC_BD": json_prompts["VersiBCB_vscc_BD"].replace("{context}", "{knowledge_doc}"),
        "VACE_BD": json_prompts["VersiBCB_vace_BD"].replace("{context}", "{knowledge_doc}"),
        # Review prompts for code review functionality
        "VSCC_REVIEW": VersiBCB_VSCC_RAG_complete_withTargetCode_v1_review_wdesc.replace("{context}", "{knowledge_doc}"),
        "VACE_REVIEW": VersiBCB_VACE_RAG_complete_withTargetCode_v1_review.replace("{context}", "{knowledge_doc}"),
        "VACE_BD_REVIEW": VersiBCB_VACE_RAG_complete_withTargetCode_v1_review.replace("{context}", "{knowledge_doc}"),
        # Error fix prompts for error fixing functionality
        "VACE_ERRORFIX": versiBCB_VACE_Prompt_errorfix,
        "VACE_BD_ERRORFIX": versiBCB_VACE_Prompt_errorfix,
        # Error fix prompts with retrieved information
        "VACE_ERRORFIX_RETRIEVE": versiBCB_VACE_Prompt_errorfix_retrieve,
        "VACE_BD_ERRORFIX_RETRIEVE": versiBCB_VACE_Prompt_errorfix_retrieve
    }
}

# dataset2prompt["versicode"]["vace"] = """
#             You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code. 
#             I will provide you with some content from the same dependency library for help.

#             Important Notes:
#             1. If the provided external knowledge (e.g., dependency documentation, code blocks, or version-specific details) is incomplete or missing, you should rely on your internal knowledge to infer the best approach for refactoring the code.
#             2. If you encounter ambiguities or uncertainties due to missing external knowledge, clearly state any assumptions you are making to proceed with the refactoring.
#             3. Your goal is to produce functional and optimized code that aligns with the new version of the dependencies, even if some external information is unavailable.

            

#             ### related content
#             {knowledge_doc}
#             ### Functionality description of the code
#             {description}

#             ### Dependency and old version
#             {origin_version}

#             ### Old version code
#             {origin_code}

#             ### Dependency and new version
#             {target_version}

#             Please only return the refactored code and enclose it with `<start>` and `<end>`.

#             ### Refactored new code
# """
# dataset2prompt["versicode"]["vscc"] = """
#             You are now a professional Python programming engineer. I will provide you with some content related to the dependency package. Then, I will also provide you with a functional description and specific dependency package version. 

#             Your task is to write Python code that implements the described functionality using the specified dependency package and version.


#             ### related content
#             {knowledge_doc}

#             Please only return the implementation code without any explanations. Enclose your code with `<start>` and `<end>` tags:

#             ### Functionality description
#             {description}

#             ### Dependency and version
#             {dependency}=={version}

# """
# dataset2prompt["versiBCB"]["vace"] = """
#             You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code. 
#             I will provide you with some content from the same dependency library for help.

#             Important Notes:
#             1. If the provided external knowledge (e.g., dependency documentation, code blocks, or version-specific details) is incomplete or missing, you should rely on your internal knowledge to infer the best approach for refactoring the code.
#             2. If you encounter ambiguities or uncertainties due to missing external knowledge, clearly state any assumptions you are making to proceed with the refactoring.
#             3. Your goal is to produce functional and optimized code that aligns with the new version of the dependencies, even if some external information is unavailable.

            

#             ### related content
#             {knowledge_doc}
#             ### Functionality description of the code
#             {description}

#             ### Dependency
#             {origin_dependency}

#             ### Old version code
#             {origin_code}

#             ### Dependency and new version
#             {target_dependency}

#             Please only return the refactored code and enclose it with `<start>` and `<end>`.

#             ### Refactored new code
# """
# dataset2prompt["versiBCB"]["vscc"] = """
#             You are now a professional Python programming engineer. I will provide you with some content related to the dependency package. Then, I will also provide you with a functional description and specific dependency package version. 

#             Your task is to write Python code that implements the described functionality using the specified dependency package and version.


#             ### related content
#             {knowledge_doc}

#             Please only return the implementation code without any explanations. Enclose your code with `<start>` and `<end>` tags:

#             ### Functionality description
#             {description}

#             ### Dependency
#             {dependency}

# """