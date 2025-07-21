import ast
import astor
import json
def clean_return_annotations(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.returns = None  # 删除返回值注解
    return astor.to_source(tree)

def clean_annotations(source: str) -> str:
    """
    删除所有变量类型注解，但保留函数返回值注解
    
    参数:
        source: 要处理的Python源代码字符串
        
    返回:
        清理后的源代码字符串
    """
    class AnnotationRemover(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            # 删除变量类型注解，保留赋值
            return ast.Assign(
                targets=[node.target],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset
            ) if node.value else None
        
        def visit_FunctionDef(self, node):
            # 保留函数定义本身，不处理返回值注解
            return self.generic_visit(node)
    
    tree = ast.parse(source)
    modified_tree = AnnotationRemover().visit(tree)
    ast.fix_missing_locations(modified_tree)
    return astor.to_source(modified_tree)
def getIDs(filename):
    item=""
    data=[]
    with open(filename, 'r') as file:
        for f in file:
            item=json.loads(f)
            data.append(item["id"])
    return data
if __name__ == "__main__":
    data=getIDs("output/baselinefix.jsonl")
    print(data)
    with open('output/temp.json', 'w') as file:
        json.dump(data, file)