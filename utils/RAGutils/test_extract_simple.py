#!/usr/bin/env python3
"""
Simple test script for extract_queries_from_response function
"""

import json
import re
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def clean_markdown_markers(text):
    """Remove markdown code block markers"""
    # Remove ```json at the beginning
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    # Remove ``` at the end
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def handle_truncated_json(json_str):
    """Handle truncated JSON by removing incomplete elements and closing properly"""
    try:
        json_str = json_str.strip()
        
        # If it doesn't end with ']', we need to reconstruct
        if not json_str.endswith(']'):
            # Strategy: Find all complete JSON objects and rebuild the array
            
            # Find complete objects by tracking brace balance
            objects = []
            current_obj_start = -1
            brace_count = 0
            in_string = False
            escape_next = False
            i = 0
            
            # Skip the opening '['
            while i < len(json_str) and json_str[i] != '{':
                i += 1
            
            if i >= len(json_str):
                return '[]'  # No objects found
            
            current_obj_start = i
            
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
                        if brace_count == 0:
                            # Found a complete object
                            obj_str = json_str[current_obj_start:i+1]
                            try:
                                # Test if it's valid JSON
                                json.loads(obj_str)
                                objects.append(obj_str)
                            except json.JSONDecodeError:
                                # Skip invalid objects
                                pass
                
                i += 1
            
            if objects:
                # Rebuild the JSON array with complete objects
                reconstructed = '[' + ','.join(objects) + ']'
                logging.info(f"Reconstructed JSON with {len(objects)} complete objects")
                return reconstructed
            else:
                return '[]'
        
        return json_str
    
    except Exception as e:
        logging.warning(f"Error handling truncated JSON: {e}")
        # Fallback: just add closing bracket
        if not json_str.endswith(']'):
            return json_str + ']'
        return json_str

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

def extract_queries_from_response(response_text):
    """Extract queries from model response as list[dict] and remove duplicates"""
    try:
        # Clean markdown code block markers
        clean_text = clean_markdown_markers(response_text)
        print(f"After cleaning markdown: {clean_text[:100]}...")
        
        # Find JSON array boundaries
        start_idx = clean_text.find('[')
        
        if start_idx == -1:
            logging.warning("No JSON array found in response")
            return []
        
        # Get everything from the start of array
        json_str = clean_text[start_idx:]
        
        # Try to parse as-is first
        try:
            parsed_data = json.loads(json_str)
            print("JSON parsed successfully without any fixes")
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Applying truncated JSON handling...")
            
            # Apply truncated handling
            json_str = handle_truncated_json(json_str)
            print(f"After reconstruction: {json_str[-100:]}")
            
            try:
                parsed_data = json.loads(json_str)
                print("JSON parsed successfully after reconstruction")
            except json.JSONDecodeError as e:
                print(f"Still failing after reconstruction: {e}")
                # Try to fix common JSON issues
                json_str = fix_common_json_issues(json_str)
                print(f"After fixes: {json_str[-100:]}")
                try:
                    parsed_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON after all fixes: {e}")
                    return []
        
        if not isinstance(parsed_data, list):
            logging.warning("Parsed JSON is not a list")
            return []
        
        # Remove duplicate dictionaries
        unique_queries = []
        seen = set()
        
        for item in parsed_data:
            if isinstance(item, dict):
                # Convert dict to a hashable representation for deduplication
                dict_str = json.dumps(item, sort_keys=True)
                if dict_str not in seen:
                    seen.add(dict_str)
                    unique_queries.append(item)
            else:
                logging.warning(f"Non-dict item found in response: {type(item)}")
        
        return unique_queries
        
    except Exception as e:
        logging.error(f"Unexpected error extracting queries: {e}")
        return []

def test_function():
    """Test the extract_queries_from_response function"""
    
    # Test case 1: Real truncated example with markdown markers
    test_input = '''```json
[
    {
        "path": "matplotlib.pyplot.subplots",
        "description": "Create a figure and a set of subplots.",
        "target_features": [
            "new layout options",
            "tight layout",
            "gridspec"
        ]
    },
    {
        "path": "numpy.random.seed",
        "description": "Seed the random number generator.",
        "target_features": [
            "new seed parameter",
            "new seed behavior"
        ]
    },
    {
        "path": "scipy.fft.fft",
        "description": "Compute the one-dimensional discrete Fourier Transform.",
        "target_features": [
            "new transform behavior",
            "new transform options"
        ]
    },
    {
        "path": "scipy.fft.fft",
        "description": "Compute the one-dimensional discrete Fourier Transform.",
        "target_features": [
            "new transform behavior",
            "new transform options"
        ]
    },
    {
        "path": "scipy.fft.fft",
        "description": "Compute the one-dimensional discrete Fourier Transform.",
        "target_features": [
            "new transform behavior",
            "new transform'''
    
    print("Testing extract_queries_from_response with truncated input...")
    print(f"Input length: {len(test_input)} characters")
    print(f"Input preview: {test_input[:100]}...")
    print()
    
    result = extract_queries_from_response(test_input)
    
    print(f"Extracted {len(result)} unique queries:")
    for i, query in enumerate(result, 1):
        print(f"{i}. {json.dumps(query, indent=2)}")
    
    print(f"\nSuccess! Processed truncated JSON with markdown markers and duplicates.")
    print(f"Original had multiple duplicates, result has {len(result)} unique items.")

if __name__ == "__main__":
    test_function() 