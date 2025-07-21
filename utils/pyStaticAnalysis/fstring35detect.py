import ast
import sys
import os
def detect_fstrings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"⚠️ Syntax error in {file_path}: {e}")
        return
    
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):  # f-string 的 AST 节点类型
            print(f"❌ F-string found in {file_path} at line {node.lineno}")
            print(f"   Code snippet: {code.splitlines()[node.lineno - 1].strip()}")
def detectFstringFromCode(code):

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"⚠️ Syntax error in {code}: {e}")
        return
    fstring_errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):  # f-string 的 AST 节点类型
            print(f"❌ F-string found in {code} at line {node.lineno}")
            print(f"   Code snippet: {code.splitlines()[node.lineno - 1].strip()}")
            fstring_errors.append(f"F-string is not supported in python 3.5. Use str.format() instead.For example, you shall replace f'{{x}} {{y}}' with '{{}} {{}}'.format(x,y) in python 3.5. Error: {code.splitlines()[node.lineno - 1].strip()}")
    return fstring_errors
def getFStringErrorInfo(generated_code,target_dependency):
    '''
        返回fstring的错误信息，以list形式返回
    '''
    if "python" in target_dependency and target_dependency["python"] == "3.5":
        fstring_errors = detectFstringFromCode(generated_code)
        return fstring_errors
    else:
        return []           
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_fstrings.py <file_or_directory>")
        sys.exit(1)
    
    target = sys.argv[1]
    if os.path.isfile(target):
        detect_fstrings(target)
    elif os.path.isdir(target):
        for root, _, files in os.walk(target):
            for file in files:
                if file.endswith('.py'):
                    detect_fstrings(os.path.join(root, file))