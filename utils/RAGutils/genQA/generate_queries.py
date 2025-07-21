#!/usr/bin/env python3
"""
Script to generate queries for RAG retrieval using a model.
This script generates queries based on the VersiBCB_VACE_RAG prompt template.
"""

import os
import json
import argparse
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmark.config.code.VersiBCB_RAGprompt import (
    VersiBCB_VACE_RAG_complete_v2,
    VersiBCB_VACE_RAG_complete_withTargetCode,
    VersiBCB_VACE_RAG_complete_withTargetCode_v1,
    VersiBCB_VACE_RAG_complete_withTargetCode_v2,
    VersiBCB_VACE_RAG_complete_withTargetCode_v3,
    VersiBCB_VACE_RAG_complete_v4,
    VersiBCB_VSCC_RAG_complete_v1,
    VersiBCB_VSCC_RAG_complete_v2,
    VersiBCB_VSCC_REVIEW
)
from utils.loraTrain.loraTrainUtils import inference
import traceback
import re

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_generation.log', mode='a'),
        logging.StreamHandler()
    ]
)

def load_data(dataset, task, ban_deprecation):
    """Load data for query generation"""
    if dataset.lower() == "versibcb":
        data_path = 'data/VersiBCB_Benchmark'
        data_name = f"{task.lower()}_datas{'_for_warning' if ban_deprecation else ''}.json"
        with open(os.path.join(data_path, data_name), "r") as f:
            datas = json.load(f)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return datas

def load_generated_code(jsonl_file_path):
    """Load generated code from jsonl file"""
    generated_code_dict = {}
    
    if not jsonl_file_path or not os.path.exists(jsonl_file_path):
        logging.warning(f"Generated code file not found or not specified: {jsonl_file_path}")
        return generated_code_dict
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'id' in data and 'answer' in data:
                        generated_code_dict[data['id']] = data['answer']
                    else:
                        logging.warning(f"Missing 'id' or 'answer' field in line {line_num}: {line}")
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON line {line_num}: {line}, error: {e}")
    
    logging.info(f"Loaded {len(generated_code_dict)} generated code items from {jsonl_file_path}")
    return generated_code_dict
def load_generated_errors(jsonl_file_path):
    """Load generated errors from jsonl file
        目前兼容json和jsonl格式
    """
    generated_errors_dict = {}
    if jsonl_file_path.endswith('.json'):
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                generated_errors_dict[item['id']] = item['error_infos']
    else:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'id' in data and 'error_infos' in data:
                            generated_errors_dict[data['id']] = data['error_infos']
                        else:
                            logging.warning(f"Missing 'id' or 'error_infos' field in line {line_num}: {line}")
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse JSON line {line_num}: {line}, error: {e}")
    return generated_errors_dict
def format_query_generation_prompt(data, task="VACE", generated_code=None, prompt_version="v2",error_info=None):
    """Format the prompt for query generation using VersiBCB_VACE_RAG template"""
    if task != "VACE" and task != "VSCC":
        logging.warning(f"Query generation currently only supports VACE and VSCC task, got {task}")
        return None
    
    # Use template with generated code if provided
    if task == "VACE":
        if generated_code is not None:
            # Select prompt template based on version
            if prompt_version == "v1":
                prompt_template = VersiBCB_VACE_RAG_complete_withTargetCode_v1
            elif prompt_version == "v2":
                prompt_template = VersiBCB_VACE_RAG_complete_withTargetCode_v2
            elif prompt_version == "v3":
                prompt_template = VersiBCB_VACE_RAG_complete_withTargetCode_v3
            else:
                prompt_template = VersiBCB_VACE_RAG_complete_withTargetCode
            
            prompt = prompt_template.format(
                description=data.get("description", ""),
                origin_dependency=data.get("origin_dependency", ""),
                origin_code=data.get("origin_code", ""),
                target_dependency=data.get("target_dependency", ""),
                generated_target_code=generated_code
            )
        else:
            # Use original template without generated code
            if prompt_version == "v2":
                prompt = VersiBCB_VACE_RAG_complete_v2.format(
                    description=data.get("description", ""),
                    origin_dependency=data.get("origin_dependency", ""),
                    origin_code=data.get("origin_code", ""),
                    target_dependency=data.get("target_dependency", "")
                )
            elif prompt_version == "v4":
                prompt = VersiBCB_VACE_RAG_complete_v4.format(
                    description=data.get("description", ""),
                    origin_dependency=data.get("origin_dependency", ""),
                    origin_code=data.get("origin_code", ""),
                    target_dependency=data.get("target_dependency", "")
                )
            else:
                raise ValueError(f"Unsupported prompt version: {prompt_version}")
    elif task == "VSCC":
        if prompt_version == "v4":
            prompt = VersiBCB_VSCC_RAG_complete_v1.format(
                description=data["description"],
                target_dependency=data["dependency"]
            )
        elif prompt_version == "v2":
            prompt = VersiBCB_VSCC_RAG_complete_v2.format(
                description=data["description"],
                target_dependency=data["dependency"]
            )
        elif prompt_version == "review":
            prompt = VersiBCB_VSCC_REVIEW.format(
                description=data["description"],
                target_dependency=data["dependency"],
                code=generated_code,
                error_info=error_info
            )
        else:
            raise ValueError(f"Unsupported prompt version: {prompt_version}")
    else:
        raise ValueError(f"Unsupported task: {task}")
    return prompt

def clean_markdown_markers(text):
    """Remove markdown code block markers and fix JSON format issues"""
    # Remove ```json at the beginning
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    # Remove ``` at the end
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    
    # Fix single quotes to double quotes for valid JSON
    # This is a simple approach - we need to be careful not to break strings
    # First, let's handle the simple case where we have {'key': 'value'} patterns
    text = re.sub(r"'([^']*)':", r'"\1":', text)  # Fix keys
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)  # Fix string values
    
    return text.strip()

def handle_truncated_json(json_str):
    """Advanced handling of truncated JSON with better reconstruction - optimized for query extraction"""
    try:
        json_str = json_str.strip()
        
        # If it doesn't end with ']', we need to reconstruct
        if not json_str.endswith(']'):
            logging.info("JSON appears to be truncated, attempting reconstruction...")
            
            # Strategy 1: Find all complete JSON objects and rebuild the array
            objects = []
            current_obj_start = -1
            brace_count = 0
            in_string = False
            escape_next = False
            i = 0
            
            # Skip the opening '['
            while i < len(json_str) and json_str[i] != '[':
                i += 1
            
            if i >= len(json_str):
                logging.warning("No opening bracket found")
                return '[]'
            
            i += 1  # Skip the opening '['
            
            while i < len(json_str):
                char = json_str[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    i += 1
                    continue
                
                if not in_string:
                    if char == '{':
                        if brace_count == 0:
                            current_obj_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and current_obj_start != -1:
                            # Found a complete object
                            obj_str = json_str[current_obj_start:i+1]
                            try:
                                # Test if it's valid JSON
                                parsed_obj = json.loads(obj_str)
                                objects.append(parsed_obj)
                                logging.debug(f"Successfully parsed object: {obj_str[:50]}...")
                            except json.JSONDecodeError:
                                logging.debug(f"Failed to parse object: {obj_str[:50]}...")
                                pass
                            current_obj_start = -1
                
                i += 1
            
            # Strategy 2: Try to fix incomplete object at the end - more aggressive completion
            if current_obj_start != -1:
                incomplete_obj = json_str[current_obj_start:]
                logging.info(f"Found incomplete object: {incomplete_obj[:100]}...")
                
                # Try to complete the incomplete object
                completed_obj = try_complete_object(incomplete_obj)
                if completed_obj:
                    objects.append(completed_obj)
                    logging.info(f"Successfully completed incomplete object")
            
            if objects:
                # Don't do deduplication here - leave it to the main function
                # Just rebuild the JSON array with complete objects
                reconstructed = json.dumps(objects, ensure_ascii=False)
                logging.info(f"Reconstructed JSON with {len(objects)} objects (before deduplication)")
                return reconstructed
            else:
                return '[]'
        
        return json_str
    
    except Exception as e:
        logging.warning(f"Error in advanced JSON reconstruction: {e}")
        return '[]'

def try_complete_object(incomplete_obj):
    """Try multiple strategies to complete an incomplete JSON object"""
    # Strategy 1: Simple quote completion
    if incomplete_obj.count('"') % 2 == 1:  # Odd number of quotes
        completion_attempts = [
            '"}',          # Complete the string and close object
            '": ""}',      # Complete as key-value pair with empty string
            '": null}',    # Complete as key-value pair with null
        ]
        
        for completion in completion_attempts:
            try:
                test_obj = incomplete_obj + completion
                parsed = json.loads(test_obj)
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Brace completion
    open_braces = incomplete_obj.count('{') - incomplete_obj.count('}')
    if open_braces > 0:
        try:
            test_obj = incomplete_obj + '}' * open_braces
            parsed = json.loads(test_obj)
            return parsed
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to find the last valid object structure
    # Look for pattern like {'path': 'something and try to complete it
    import re
    path_pattern = r"\{'path':\s*'([^']*)"
    match = re.search(path_pattern, incomplete_obj)
    if match:
        partial_path = match.group(1)
        # Create a completed object with the partial path
        try:
            completed = {'path': partial_path}
            return completed
        except:
            pass
    
    return None

def fix_common_json_issues(json_str):
    """Fix common JSON formatting issues"""
    try:
        # Remove trailing commas before closing braces and brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Ensure the JSON ends with a closing bracket
        if not json_str.strip().endswith(']'):
            json_str = json_str.strip() + ']'
        
        return json_str
    
    except Exception as e:
        logging.warning(f"Error fixing JSON issues: {e}")
        return json_str

def extract_individual_objects(text):
    """Extract individual JSON objects from text when array parsing fails"""
    objects = []
    
    # Pattern to find JSON-like objects with nested structures
    # This pattern handles simple nested objects but not deeply nested ones
    object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    matches = re.finditer(object_pattern, text)
    
    for match in matches:
        obj_str = match.group()
        try:
            parsed_obj = json.loads(obj_str)
            objects.append(parsed_obj)
            logging.debug(f"Extracted individual object: {obj_str[:100]}...")
        except json.JSONDecodeError:
            continue
    
    logging.info(f"Extracted {len(objects)} individual objects")
    return objects

def extract_queries_from_response(response_text):
    """Extract queries from model response as list[dict] and remove duplicates"""
    try:
        # Clean markdown code block markers
        clean_text = clean_markdown_markers(response_text)
        
        # Find JSON array boundaries
        start_idx = clean_text.find('[')
        
        if start_idx == -1:
            logging.warning("No JSON array found in response")
            # Try to find individual JSON objects
            return extract_individual_objects(clean_text)
        
        # Get everything from the start of array
        json_str = clean_text[start_idx:]
        
        # Try to parse as-is first
        try:
            parsed_data = json.loads(json_str)
            logging.info("JSON parsed successfully on first attempt")
        except json.JSONDecodeError as e:
            logging.info(f"JSON parsing failed, applying advanced reconstruction: {e}")
            
            # Apply advanced truncated handling
            json_str = handle_truncated_json(json_str)
            
            try:
                parsed_data = json.loads(json_str)
                logging.info("JSON parsed successfully after advanced reconstruction")
            except json.JSONDecodeError as e:
                logging.warning(f"Still failing after advanced reconstruction: {e}")
                # Last resort: try to extract individual objects
                return extract_individual_objects(clean_text)
        
        if not isinstance(parsed_data, list):
            logging.warning("Parsed JSON is not a list")
            if isinstance(parsed_data, dict):
                return [parsed_data]
            return []
        
        # Remove exact duplicate dictionaries, but preserve legitimate repeats
        unique_queries = []
        seen_exact = set()
        
        for item in parsed_data:
            if isinstance(item, dict):
                # Convert dict to a hashable representation for exact deduplication only
                dict_str = json.dumps(item, sort_keys=True)
                if dict_str not in seen_exact:
                    seen_exact.add(dict_str)
                    unique_queries.append(item)
                else:
                    logging.debug(f"Skipping exact duplicate: {item}")
            elif isinstance(item, str):
                # Handle string items
                if item not in seen_exact:
                    seen_exact.add(item)
                    unique_queries.append({'description': item})
            else:
                logging.warning(f"Unexpected item type found in response: {type(item)}")
        
        # Log detailed extraction results
        original_count = len(parsed_data)
        final_count = len(unique_queries)
        logging.info(f"Extracted {final_count} unique queries from {original_count} total items")
        
        if final_count < original_count:
            logging.info(f"Removed {original_count - final_count} exact duplicates")
        
        return unique_queries
        
    except Exception as e:
        logging.error(f"Unexpected error extracting queries: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return []

def generate_queries_for_data(data_list, model, tokenizer, args, generated_code_dict=None, temp_output_path=None,error_infos_dict=None):
    """Generate queries for a list of data items"""
    results = {}
    
    for i, data_item in enumerate(tqdm(data_list, desc="Generating queries")):
        sample_id = data_item.get('id', f'index_{i}')
        
        try:
            # Get generated code if available and enabled
            generated_code = None
            if args.enable_generated_code and generated_code_dict:
                generated_code = generated_code_dict.get(sample_id)
                if generated_code is None:
                    logging.warning(f"No generated code found for sample {sample_id}, using original template")
            if error_infos_dict:
                error_info = error_infos_dict.get(sample_id)
            else:
                error_info = None
            # Format prompt for query generation
            prompt = format_query_generation_prompt(
                data_item, 
                args.task, 
                generated_code=generated_code,
                prompt_version=getattr(args, 'prompt_version', 'v1'),
                error_info=error_info
            )
            if prompt is None:
                continue
                
            # Generate queries using the model
            response_text = inference(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                inference_type=args.inference_type,
                api_key=args.api_key,
                model_name=args.api_model_name,
                api_base_url=args.huggingface_api_base_url
            )
            
            # Extract queries from response
            queries = extract_queries_from_response(response_text)
            
            if not queries:
                logging.warning(f"No valid queries extracted for sample {sample_id}")
                queries = [data_item.get("description", "")]  # Fallback to description
            
            results[sample_id] = {
                "queries": queries,
                "raw_response": response_text,
                "original_data": {
                    "description": data_item.get("description", ""),
                    "origin_dependency": data_item.get("origin_dependency", ""),
                    "target_dependency": data_item.get("target_dependency", "") if "target_dependency" in data_item else data_item["dependency"]
                }
            }
            # 临时不断存放
            save_queries(results, temp_output_path)
            logging.info(f"Generated {len(queries)} queries for sample {sample_id}")
            
        except Exception as e:
            logging.error(f"Error generating queries for sample {sample_id}: {e}")
            logging.error(traceback.format_exc())
            # Fallback to original description
            results[sample_id] = {
                "queries": [data_item.get("description", "")],
                "raw_response": "",
                "original_data": {
                    "description": data_item.get("description", ""),
                    "origin_dependency": data_item.get("origin_dependency", ""),
                    "target_dependency": data_item.get("target_dependency", "") if "target_dependency" in data_item else data_item["dependency"]
                },
                "error": str(e)
            }
            
    return results

def load_existing_queries(output_path):
    """Load existing queries if file exists"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load existing queries from {output_path}: {e}")
    return {}

def save_queries(queries_dict, output_path):
    """Save generated queries to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(queries_dict, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved queries to {output_path}")

def load_target_task_ids(target_task_ids_file, target_ids_str):
    """Load target task IDs from file or direct string specification"""
    # Priority: direct string > file
    if target_ids_str is not None:
        try:
            # Parse comma-separated task IDs
            target_ids = [id_str.strip() for id_str in target_ids_str.split(',') if id_str.strip()]
            target_ids = [int(id_str) for id_str in target_ids]
            logging.info(f"Loaded {len(target_ids)} target task IDs from command line: {target_ids}")
            return set(target_ids)
        except Exception as e:
            logging.error(f"Error parsing target task IDs from command line: {e}")
            return None
    
    if target_task_ids_file is not None:
        try:
            with open(target_task_ids_file, 'r') as f:
                target_task_ids = json.load(f)
            logging.info(f"Loaded {len(target_task_ids)} target task IDs from file {target_task_ids_file}")
            return set(target_task_ids) if isinstance(target_task_ids, list) else target_task_ids
        except Exception as e:
            logging.error(f"Error loading target task IDs from file: {e}")
            return None
    
    return None

def filter_and_distribute_data(data_list, target_task_ids, rank, world_size, start_idx=0, end_idx=None):
    """Filter data by target task IDs and distribute across workers"""
    # Apply index filtering first
    if end_idx is not None:
        data_list = data_list[start_idx:end_idx]
    elif start_idx > 0:
        data_list = data_list[start_idx:]
    
    # Filter by target task IDs if specified
    filtered_data = []
    skipped_by_task_id_filter = 0
    
    for i, data_item in enumerate(data_list):
        sample_id = data_item.get('id', f'index_{i + start_idx}')
        
        # Check if sample is in target task IDs (if specified)
        if target_task_ids is not None:
            if sample_id not in target_task_ids:
                skipped_by_task_id_filter += 1
                continue
        
        filtered_data.append(data_item)
    
    # Distribute data across workers using interleaved allocation
    worker_data = []
    for idx in range(rank, len(filtered_data), world_size):
        worker_data.append(filtered_data[idx])
    
    logging.info(f"Worker {rank}: Total data after filtering: {len(filtered_data)}")
    if target_task_ids is not None:
        logging.info(f"Worker {rank}: Skipped {skipped_by_task_id_filter} samples due to task ID filtering")
    logging.info(f"Worker {rank}: Assigned {len(worker_data)} samples using interleaved allocation")
    
    return worker_data

def main():
    parser = argparse.ArgumentParser(description="Generate queries for RAG retrieval using a model")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="VersiBCB", help="Dataset name")
    parser.add_argument("--task", type=str, default="VACE", help="Task type")
    parser.add_argument("--ban_deprecation", action="store_true", help="Whether to ban deprecation")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name or path")
    parser.add_argument("--inference_type", type=str, default="local", choices=['local', 'huggingface', 'togetherai'], help="Inference type")
    parser.add_argument("--precision", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'], help="Model precision")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    
    # API arguments (for remote inference)
    parser.add_argument("--api_key", type=str, default=None, help="API key for remote inference")
    parser.add_argument("--api_model_name", type=str, default=None, help="Model name for API")
    parser.add_argument("--huggingface_api_base_url", type=str, default=None, help="HuggingFace API base URL")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="generated_queries", help="Output directory for generated queries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing queries")
    
    # Processing arguments
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing")
    
    # Multi-worker arguments
    parser.add_argument("--rank", type=int, default=0, help="Worker rank for distributed processing")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    
    # Target task IDs filtering
    parser.add_argument("--target_task_ids_file", type=str, default=None,
                       help="Path to file containing target task IDs (JSON format). Only these tasks will be processed.")
    parser.add_argument("--target_ids", type=str, default=None,
                       help="Comma-separated list of target task IDs (e.g., '92,98,223'). Only these tasks will be processed.")
    # 是否启用output_code(会启用另一个prompt进行query生成),以及output code对应的位置
    #TODO:修改该名，取得不太好
    parser.add_argument("--enable_generated_code", action="store_true", default=False,
                       help="Enable output code for query generation")
    parser.add_argument("--generated_code_path", type=str, default=None,
                       help="Path to the output code file")
    parser.add_argument("--prompt_version","-pv", type=str, default="v3", choices=["v0", "v1", "v2","v3","v4","review"],
                       help="Version of the prompt template to use when enable_output_code is True")
    parser.add_argument("--output_filename", type=str, default=None,
                       help="Output filename")
    parser.add_argument("--generated_errors_path", type=str, default=None,
                       help="Path to the generated errors file")
    args = parser.parse_args()
    
    rank = args.rank
    world_size = args.world_size
    
    # Configure logging for multi-worker
    log_level = logging.INFO
    log_format = f'%(asctime)s - Worker {rank} - %(levelname)s - %(message)s'
    
    # Reconfigure logging for this worker
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(f'query_generation_worker_{rank}.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Worker {rank}/{world_size}: Starting query generation")
    
    # Load data
    logging.info(f"Worker {rank}: Loading data from {args.dataset} dataset, task: {args.task}")
    data_list = load_data(args.dataset, args.task, args.ban_deprecation)
    logging.info(f"Worker {rank}: Loaded {len(data_list)} total data items")
    
    # Load generated code if enabled
    generated_code_dict = None
    if args.enable_generated_code:
        if args.generated_code_path:
            logging.info(f"Worker {rank}: Loading generated code from {args.generated_code_path}")
            generated_code_dict = load_generated_code(args.generated_code_path)
            logging.info(f"Worker {rank}: Using prompt version {args.prompt_version} for queries with generated code")
        else:
            logging.warning(f"Worker {rank}: enable_generated_code is True but generated_code_path is not provided")
            args.enable_generated_code = False
    if args.generated_errors_path:
        error_infos_dict = load_generated_errors(args.generated_errors_path)
    # Load target task IDs
    target_task_ids = load_target_task_ids(args.target_task_ids_file, args.target_ids)
    
    # Filter and distribute data across workers
    worker_data = filter_and_distribute_data(
        data_list, target_task_ids, rank, world_size, args.start_idx, args.end_idx
    )
    
    if len(worker_data) == 0:
        logging.info(f"Worker {rank}: No data to process. Exiting.")
        return
    
    # Determine output path with worker-specific naming for intermediate results
    code_suffix = f"_with_code_{args.prompt_version}" if args.enable_generated_code else ""
    if args.output_filename:
        output_filename = args.output_filename
    else:
        output_filename = f"{args.dataset.lower()}_{args.task.lower()}_queries{'_BD' if args.ban_deprecation else ''}{code_suffix}.json"
    
    if world_size > 1:
        # For multi-worker, create temporary worker-specific files with same suffix as final
        temp_output_filename = f"{args.dataset.lower()}_{args.task.lower()}_queries{'_BD' if args.ban_deprecation else ''}{code_suffix}_worker_{rank}.json"
        temp_output_path = os.path.join(args.output_dir, temp_output_filename)
        final_output_path = os.path.join(args.output_dir, output_filename)
    else:
        # Single worker uses final output path directly
        temp_output_path = os.path.join(args.output_dir, output_filename)
        final_output_path = temp_output_path
    
    # Load existing queries (only for the final output file)
    existing_queries = {}
    if not args.overwrite and os.path.exists(final_output_path):
        existing_queries = load_existing_queries(final_output_path)
        logging.info(f"Worker {rank}: Loaded {len(existing_queries)} existing queries")
    
    # Filter data that already has queries
    data_to_process = []
    for data_item in worker_data:
        sample_id = data_item.get('id', f'index_{data_list.index(data_item)}')
        if sample_id not in existing_queries:
            data_to_process.append(data_item)
    
    logging.info(f"Worker {rank}: Need to generate queries for {len(data_to_process)} new items")
    
    if len(data_to_process) == 0:
        logging.info(f"Worker {rank}: No new data to process. Exiting.")
        return
    
    # Initialize model for local inference
    model, tokenizer = None, None
    if args.inference_type == "local":
        logging.info(f"Worker {rank}: Loading local model: {args.model}")
        
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        precision = precision_map.get(args.precision, torch.float16)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=precision
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        logging.info(f"Worker {rank}: Model loaded with device map: {model.hf_device_map}")
    
    # Generate queries
    logging.info(f"Worker {rank}: Starting query generation...")
    new_queries = generate_queries_for_data(data_to_process, model, tokenizer, args, generated_code_dict,temp_output_path,error_infos_dict)
    
    # Save worker results to temporary file
    save_queries(new_queries,temp_output_path)
    
    logging.info(f"Worker {rank}: Query generation completed. Generated {len(new_queries)} new queries.")
    logging.info(f"Worker {rank}: Results saved to temporary file: {temp_output_path}")
    if world_size > 1:
        logging.info(f"Worker {rank}: Final results will be merged to: {final_output_path}")
    else:
        logging.info(f"Worker {rank}: Final results saved to: {final_output_path}")
    
    # Print some statistics
    query_counts = [len(q_data.get("queries", [])) for q_data in new_queries.values()]
    if query_counts:
        avg_queries = sum(query_counts) / len(query_counts)
        logging.info(f"Worker {rank}: Average queries per sample: {avg_queries:.2f}")
        logging.info(f"Worker {rank}: Min queries per sample: {min(query_counts)}")
        logging.info(f"Worker {rank}: Max queries per sample: {max(query_counts)}")
    
    # For multi-worker, rank 0 will merge all results
    if world_size > 1 and rank == 0:
        logging.info(f"Worker {rank}: Waiting for other workers to complete...")
        import time
        time.sleep(5)  # Give other workers time to finish
        
        # Merge all worker results
        merged_queries = {}
        if not args.overwrite and os.path.exists(final_output_path):
            merged_queries = load_existing_queries(final_output_path)
        
        for worker_rank in range(world_size):
            worker_file = os.path.join(
                args.output_dir, 
                f"{args.dataset.lower()}_{args.task.lower()}_queries{'_BD' if args.ban_deprecation else ''}{code_suffix}_worker_{worker_rank}.json"
            )
            
            if os.path.exists(worker_file):
                worker_queries = load_existing_queries(worker_file)
                merged_queries.update(worker_queries)
                logging.info(f"Worker {rank}: Merged {len(worker_queries)} queries from worker {worker_rank}")
                
                # Clean up temporary file
                os.remove(worker_file)
            else:
                logging.warning(f"Worker {rank}: Worker {worker_rank} output file not found: {worker_file}")
        
        # Save final merged results
        save_queries(merged_queries, final_output_path)
        logging.info(f"Worker {rank}: Final merged results saved to: {final_output_path}")
        logging.info(f"Worker {rank}: Total merged queries: {len(merged_queries)}")
        
        # Final statistics
        final_query_counts = [len(q_data.get("queries", [])) for q_data in merged_queries.values()]
        if final_query_counts:
            final_avg_queries = sum(final_query_counts) / len(final_query_counts)
            logging.info(f"Worker {rank}: Final average queries per sample: {final_avg_queries:.2f}")
            logging.info(f"Worker {rank}: Final min queries per sample: {min(final_query_counts)}")
            logging.info(f"Worker {rank}: Final max queries per sample: {max(final_query_counts)}")
    
    elif world_size == 1:
        # Single worker case
        merged_queries = {**existing_queries, **new_queries}
        save_queries(merged_queries, final_output_path)
        logging.info(f"Worker {rank}: Single worker results saved to: {final_output_path}")
        logging.info(f"Worker {rank}: Total queries: {len(merged_queries)}")
    
    logging.info(f"Worker {rank}: Task completed")
# def assign_target_dependency(data):
class QueryFilter:
    def __init__(self, data):
        self.data = data

    def filter_queries(self, queries,target_dependency=None):
        """
        Filter queries based on target_dependency.
        Only keep queries where the package of target_api exists in target_dependency.
        
        Args:
            queries: List of query dictionaries, each containing 'query' and 'target_api'
            
        Returns:
            List of filtered queries
        """
        if not queries:
            return []
        if type(queries) == str:
            return []
        if type(queries) == list and "target_api" not in queries[0]:
            return []
        # Get target_dependency from the data
        # i
        # target_dependency = self.data.get("original_data", {}).get("target_dependency", {})
        
        # If no target_dependency, return all queries
        if not target_dependency:
            return queries
        
        filtered_queries = []
        
        for query in queries:
            
            target_api = query.get("target_api", "")
            if not target_api:
                continue
                
            # Extract package name from target_api (first part before the first dot)
            package_name = target_api.split(".")[0]
            
            # Check if the package exists in target_dependency
            if package_name in target_dependency and package_name!='python':
                filtered_queries.append(query)
        
        return filtered_queries
    def filterTargetFile(self, target_file,output_file=None):
        '''
            根据target_file中的target_dependency进行过滤
        '''
        with open(target_file, "r") as f:
            target_queries = json.load(f)
        for k, v in target_queries.items():
            target_queries[k]["queries"] = self.filter_queries(v["queries"],v["original_data"]["target_dependency"])
        if output_file:
            with open(output_file, "w") as f:
                json.dump(target_queries, f, indent=2, ensure_ascii=False)
        return target_queries
        
def getID2dependency(vscc_datas):
    id2dependency = {}
    for vscc_data in vscc_datas:
        id2dependency[str(vscc_data["id"])] = vscc_data["dependency"]
    return id2dependency

def tempfix():
    '''
        解决结果中缺乏target_dependency的问题
    '''
    with open("data/generated_queries/versibcb_vscc_queries.json", "r") as f:
        data = json.load(f)
    # 根据vscc数据的相同id对于target_dependency进行赋值
    with open("data/VersiBCB_Benchmark/vscc_datas.json", "r") as f:
        vscc_datas = json.load(f)
    id2dependency = getID2dependency(vscc_datas)
    for k, v in data.items():
        if k in id2dependency:
            data[k]["original_data"]["target_dependency"] = id2dependency[k]
    with open("data/generated_queries/versibcb_vscc_queries_with_target_dependency.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    pass
if __name__ == "__main__":
    # tempfix()
    # main()
    query_filter = QueryFilter(None)
    query_filter.filterTargetFile("data/generated_queries/versibcb_vscc_queries_with_code_review.json", "data/generated_queries/versibcb_vscc_queries_with_code_review_filtered.json")


