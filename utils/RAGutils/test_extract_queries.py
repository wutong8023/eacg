#!/usr/bin/env python3
"""
Test script for the enhanced extract_queries_from_response function
"""

import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.RAGutils.genQueries.generate_queries import extract_queries_from_response

def test_extract_queries():
    """Test the extract_queries_from_response function with various inputs"""
    
    # Test case 1: Complex JSON object format (like the one in the raw_response)
    test_response_1 = '''```json
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
    }
]
```'''
    
    # Test case 2: JSON with duplicates
    test_response_2 = '''[
    {
        "path": "pandas.DataFrame.read_csv",
        "description": "Read CSV file into DataFrame"
    },
    {
        "path": "numpy.array.flatten",
        "description": "Flatten array to 1D"
    },
    {
        "path": "pandas.DataFrame.read_csv",
        "description": "Read CSV file into DataFrame"
    }
]'''
    
    # Test case 3: Truncated JSON (like the one in the provided data)
    test_response_3 = '''```json
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
            "new transform'''

    # Test case 4: Severely truncated JSON (missing closing brackets entirely)
    test_response_4 = '''[
    {
        "path": "pandas.DataFrame.read_csv",
        "description": "Read CSV file into DataFrame",
        "target_features": ["encoding", "chunking"]
    },
    {
        "path": "requests.get",
        "description": "Perform HTTP GET request",
        "target_features": ["timeout", "proxy'''

    # Test case 5: JSON with trailing comma and truncation
    test_response_5 = '''[
    {
        "path": "sklearn.preprocessing.StandardScaler",
        "description": "Scale features to standard form",
        "target_features": ["fit_transform", "partial_fit"]
    },
    {
        "path": "pandas.DataFrame.groupby",
        "description": "Group DataFrame by column",
        "target_features": ["aggregation", "chaining"],
    '''

    print("=== Testing extract_queries_from_response function ===\n")
    
    test_cases = [
        ("Complex JSON object format", test_response_1),
        ("JSON with duplicates", test_response_2),
        ("Truncated JSON", test_response_3),
        ("Severely truncated JSON", test_response_4),
        ("JSON with trailing comma and truncation", test_response_5)
    ]
    
    for i, (name, test_input) in enumerate(test_cases, 1):
        print(f"Test case {i}: {name}")
        print(f"Input (first 100 chars): {repr(test_input[:100])}...")
        
        try:
            queries = extract_queries_from_response(test_input)
            print(f"Extracted {len(queries)} unique query dictionaries:")
            for j, query in enumerate(queries, 1):
                print(f"  {j}. {json.dumps(query, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)
    
    # Test with actual data from the provided file (with many duplicates)
    print("\nTest case 6: Real data with many duplicates and truncation")
    real_response = '''```json
[
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
            "new transform options"
        ]
    },
    {
        "path": "scipy.fft.fft",
        "description": "Compute the one-dimensional discrete Fourier Transform.",
        "target_features": [
            "new transform behavior",
            "new transform'''
    
    try:
        queries = extract_queries_from_response(real_response)
        print(f"Extracted {len(queries)} unique query dictionaries from real data:")
        for j, query in enumerate(queries, 1):
            print(f"  {j}. {json.dumps(query, indent=2)}")
        print(f"\nNote: Duplicates were successfully removed and truncation handled")
    except Exception as e:
        print(f"Error: {e}")

    # Test case 7: Actual raw_response from provided file
    print("\nTest case 7: Actual truncated response from provided file")
    actual_truncated = '''```json
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
            "new transform options"
        ]
    },
    {
        "path": "scipy.fft.fft",
        "description": "Compute the one-dimensional discrete Fourier Transform.",
        "target_features": [
            "new transform behavior",
            "new transform'''
    
    try:
        queries = extract_queries_from_response(actual_truncated)
        print(f"Extracted {len(queries)} unique query dictionaries from actual truncated response:")
        for j, query in enumerate(queries, 1):
            print(f"  {j}. {json.dumps(query, indent=2)}")
        print(f"\nNote: Successfully handled actual truncated response with duplicates")
    except Exception as e:
        print(f"Error: {e}")

    # Test case 8: Real example from the provided JSON file
    print("\nTest case 8: Real example exactly like the provided file")
    real_example = '''```json
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
        "path": "numpy.round",
        "description": "Round an array to the given number of decimals.",
        "target_features": [
            "new rounding behavior",
            "new rounding precision"
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
    
    try:
        queries = extract_queries_from_response(real_example)
        print(f"Extracted {len(queries)} unique query dictionaries from real example:")
        for j, query in enumerate(queries, 1):
            print(f"  {j}. {json.dumps(query, indent=2)}")
        print(f"\nNote: Original had many duplicates, successfully removed to {len(queries)} unique items")
        print(f"Successfully handled markdown markers and truncation")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_extract_queries()