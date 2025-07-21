import json
import argparse
from pathlib import Path

def is_repetitive_content(text: str, threshold: int = 3) -> bool:
    """
    Check if the text contains repetitive content.
    Returns True if the same sentence appears more than threshold times.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return False
    
    # Check if the same sentence appears multiple times
    for sentence in sentences:
        if sentences.count(sentence) > threshold:
            return True
    return False

def is_valid_target_api(target_api: str, dependencies: dict) -> bool:
    """
    Check if the first part of target_api exists in dependencies keys.
    """
    if not target_api or not dependencies:
        return False
    
    # Get the first part of target_api (before the first dot)
    api_package = target_api.split('.')[0]
    
    # Check if the package exists in dependencies
    return api_package in dependencies

def modify_equivalent_answers(input_file: str, output_file: str):
    """
    Modify answers in the JSONL file based on specific conditions:
    1. query contains 'equivalent'
    2. retrieval_method is 'exact_api_match'
    3. target_api exists in query
    4. first part of target_api exists in dependencies
    
    Also removes items with:
    1. CUDA out of memory errors
    2. Repetitive content
    3. Invalid target_api (not in dependencies)
    """
    modified_count = 0
    total_count = 0
    removed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_count += 1
            item = json.loads(line.strip())
            
            # Skip items with CUDA out of memory errors
            if 'answer' in item and 'Error generating answer: CUDA out of memory' in item['answer']:
                removed_count += 1
                continue
                
            # Skip items with repetitive content
            if 'answer' in item and is_repetitive_content(item['answer']):
                removed_count += 1
                continue
            
            # Skip items where target_api is not in dependencies
            if not is_valid_target_api(item.get('target_api', ''), item.get('dependencies', {})):
                removed_count += 1
                continue
            
            # Check all conditions for equivalent API
            if ('query' in item and 'equivalent' in item['query'].lower() and 
                item['retrieval_method'] == 'exact_api_match' and 
                item['target_api'] in item['query']):
                
                # Modify the answer to just the target_api
                item['answer'] = item['target_api']
                modified_count += 1
            
            # Write the item (modified or not) to output file
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Total items processed: {total_count}")
    print(f"Items modified: {modified_count}")
    print(f"Items removed: {removed_count}")

def main():
    parser = argparse.ArgumentParser(description='Modify answers in JSONL file based on specific conditions')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modify_equivalent_answers(args.input_file, args.output_file)

if __name__ == '__main__':
    main() 