versiBCB_VACE_Prompt = '''
You are an expert Python programmer. Please edit the target code to fix the errors while reserving the original functionality.
### task requirement
{task_requirement}
### dependency version the code need to run on
{target_dependency}
### target code
{target_code}
Below is detected errors detected in the target code.
{error_info}

Now, fix the error given the error info and the dependency version. 
### fixed target code(wrapped by '```python' and '```')

'''