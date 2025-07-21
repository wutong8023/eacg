VersiBCB_VACE_RAG_instruct='''
You are an expert Python refactoring engineer. Your goal is to identify APIs and functionalities that require refactoring when upgrading Python dependencies.

I will provide you with:
1.  **A description of the code's functionality.**
2.  **The original code snippet, including its dependencies and their versions.**
3.  **The target dependencies with their new versions.**

Your task is to generate a list of queries to retrieve information from a RAG database. These queries should focus on APIs or functionalities whose usage might have changed or that you are unfamiliar with in the context of the `target_dependency` versions.



---

### Query Generation Guidelines

For each query, use one of the following formats:

* **API Path Query:** Use `path:` followed by the full API path (e.g., `path:pandas.DataFrame.read_csv`, `path:numpy.array.flatten`).
* **Functionality Description Query:** Use `description:` followed by a concise description of a sub-functionality (e.g., `description:calculate the mean of column`, `description:handle missing values in dataframe`).

**Important:** Only generate queries for APIs or functionalities that you are **unsure about** or suspect require changes due to the new dependency versions. Focus on specific changes or potential incompatibilities.

---
---

### Provided Information

**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

---

Now, generate the query list given above information.

### True Generated Query List
'''
VersiBCB_VACE_RAG_complete='''
You are an expert Python refactoring engineer. Your goal is to identify APIs and functionalities that require refactoring when upgrading Python dependencies.

I will provide you with:
1.  **A description of the code's functionality.**
2.  **The original code snippet, including its dependencies and their versions.**
3.  **The target dependencies with their new versions.**

Your task is to generate a list of queries to retrieve information from a RAG database. These queries should focus on APIs or functionalities whose usage might have changed or that you are unfamiliar with in the context of the `target_dependency` versions.



---

### Query Generation Guidelines

For each query, use one of the following formats:

* **API Path Query:** Use `path:` followed by the full API path (e.g., `{{path:pandas.DataFrame.read_csv}}`, `{{path:numpy.array.flatten}}`).
* **Functionality Description Query:** Use `description:` followed by a concise description of a sub-functionality (e.g., {{description:calculate the mean of column}}`, `{{description:handle missing values in dataframe}}`).
* **API Path Query along with Description Query:** Use `path:description:` followed by the full API path and a concise description of a sub-functionality (e.g., {{'path':'pandas.DataFrame.read_csv', 'description':'read a csv file'}}).

**Important:** Only generate queries for APIs or functionalities that you are **unsure about** or suspect require changes due to the new dependency versions. Focus on specific changes or potential incompatibilities.

---
---

### Provided Information

**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

---

Now, generate the query list given above information.

### Generated Query List to retrieve context for code refactoring
'''
VersiBCB_VACE_RAG_complete_v1='''
You are an expert Python refactoring engineer. Your primary goal is to identify and query for information on APIs and functionalities that require changes when upgrading Python dependencies.

I will provide you with:
1.  **A detailed description of the code's functionality.**
2.  **The original code snippet, including all its dependencies and their specific versions.**
3.  **The target dependencies with their new, upgraded versions.**

Your task is to generate a list of precise queries that will be used to retrieve relevant information from a RAG (Retrieval Augmented Generation) database. These queries must specifically target APIs or functionalities where the usage or behavior might have changed, or that you are unfamiliar with, in the context of the `target_dependency` versions.

---

### Query Format Guidelines

For each query, use one of the following JSON formats. Ensure the output is a valid JSON list.

* **1. API Path Query (Single String):**
    Use this when you need information solely about an API's path.
    Example: ` {{path:pandas.DataFrame.read_csv}} `
    Example: ` {{path:numpy.array.flatten}} `

* **2. Functionality Description Query (Single String):**
    Use this when you need information solely about a sub-functionality.
    Example: ` {{description:calculate the mean of column in DataFrame}} `
    Example: ` {{description:handle missing values in DataFrame}} `

* **3. Combined API Path, Description, and Target Features Query (JSON Object):**
    Use this when you need to query a specific API, provide a description about its core functionality (to help retrieve information even if the exact API path has changed), and also specify the **exact features or aspects you are looking for** within its documentation.
    Example: ` {{"path": "pandas.DataFrame.read_csv", "description": "Functionality for loading tabular data from CSV files.", "target_features": ["new encoding options", "chunking", "date parsing"]}} `
    Example: ` {{"path": "requests.get", "description": "Performing HTTP GET requests to retrieve web content.", "target_features": ["timeout parameter changes", "proxy configuration", "authentication methods"]}} `
    Example: ` {{"path": "sklearn.preprocessing.StandardScaler", "description": "Data scaling and standardization transform for machine learning.", "target_features": ["fit_transform method", "partial_fit for large datasets", "inverse_transform behavior"]}} `

**Crucial Rule:** Only generate queries for APIs or functionalities that you are **uncertain about** or strongly **suspect require modification** due to the new dependency versions. Prioritize queries that address specific changes, deprecations, or potential incompatibilities. Do not generate queries for unchanged or clearly understood parts of the code. 

---

### Provided Information for Refactoring

**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

---

Now, generate the query list based on the instructions above.

### Generated Query List to retrieve context for code refactoring(only public method,class,function,attribute,property for target_dependency {target_dependency} should be included, python builtin package should be excluded, like subprocess, os, time, etc.)
'''
# 更好的description，方便与docstring进行相似度匹配
VersiBCB_VACE_RAG_complete_v2='''
You are an expert Python refactoring engineer. Your primary goal is to identify and query for information on APIs and functionalities that require changes when upgrading Python dependencies.

I will provide you with:
1.  **A detailed description of the code's functionality.**
2.  **The original code snippet, including all its dependencies and their specific versions.**
3.  **The target dependencies with their new, upgraded versions.**

Your task is to generate a list of precise queries that will be used to retrieve relevant information from a RAG (Retrieval Augmented Generation) database. These queries must specifically target APIs or functionalities where the usage or behavior might have changed, or that you are unfamiliar with, in the context of the `target_dependency` versions.

---

### Query Format Guidelines

For each query, use one of the following JSON formats. Ensure the output is a valid JSON list.

* **1. API Path Query (Single String):**
    Use this when you need information solely about an API's path.
    Example: ` {{path:pandas.DataFrame.read_csv}} `
    Example: ` {{path:numpy.array.flatten}} `

* **2. Functionality Description Query (Single String):**
    Use this when you need information solely about a sub-functionality.
    Example: ` {{description:calculate the mean of column in DataFrame}} `
    Example: ` {{description:handle missing values in DataFrame}} `

* **3. Combined API Path, Description, and Target Features Query (JSON Object):**
    Use this when you need to query a specific API, provide a description about its core functionality,parameter (to help retrieve information even if the exact API path has changed), and also specify the **exact features or aspects you are looking for** within its documentation.
    Example: ` {{'path': 'pandas.DataFrame.read_csv', 'description': 'Reads a CSV file into a DataFrame. Parameters include `filepath_or_buffer` (str, Path, or file-like object) and `sep` (str, default \',\'). Returns a new `DataFrame`.', 'target_features': ['new encoding options', 'chunking', 'date parsing']}} `

    Example: ` {{'path': 'requests.get', 'description': 'Performs an HTTP GET request. Takes `url` (str) as the primary parameter and optional parameters like `params` (dict), `headers` (dict), `timeout` (float or tuple). Returns a `Response` object.', 'target_features': ['timeout parameter changes', 'proxy configuration', 'authentication methods']}} `

    Example: ` {{'path': 'sklearn.preprocessing.StandardScaler', 'description': 'Scales features to zero mean and unit variance. `fit` method accepts array-like `X`, `transform` method also accepts `X` and returns a scaled `ndarray`.', 'target_features': ['fit_transform method', 'partial_fit for large datasets', 'inverse_transform behavior']}} `

**Crucial Rule:** Only generate queries for APIs or functionalities that you are **uncertain about** or strongly **suspect require modification** due to the new dependency versions. Prioritize queries that address specific changes, deprecations, or potential incompatibilities. Do not generate queries for unchanged or clearly understood parts of the code. 

---

### Provided Information for Refactoring

**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

---

Now, generate the query list based on the instructions above.

### Generated Query List to retrieve context for code refactoring(only public method,class,function,attribute,property for target_dependency {target_dependency} should be included, python builtin package should be excluded, like subprocess, os, time, etc.)
'''
VersiBCB_VACE_RAG_complete_withTargetCode = '''
You are an expert Python code migration engineer. Your primary goal is to identify and query for information on APIs and functionalities that require changes when Python dependencies migrates.

I will provide you with:
1.  **A detailed description of the code's functionality.**
2.  **The original code snippet, including all its dependencies and their specific versions.**
3.  **The target dependencies with their new, upgraded versions.**
4.  **The generated target code.**

Your task is to generate a list of precise queries that will be used to retrieve relevant information from a RAG (Retrieval Augmented Generation) database for target code revision. These queries must specifically target APIs or functionalities where the usage or behavior might have changed, or that you are unfamiliar with, in the context of the `target_dependency` versions.

---

### Query Format Guidelines

For each query, use one of the following JSON formats. Ensure the output is a valid JSON list.

* **1. API Path Query (Single String):**
    Use this when you need information solely about an API's path.
    Example: ` {{'path':'pandas.DataFrame.read_csv'}} `
    Example: ` {{'path':'numpy.array.flatten'}} `

* **2. Functionality Description Query (Single String):**
    Use this when you need information solely about a sub-functionality.
    Example: ` {{'description':'calculate the mean of column in DataFrame'}} `
    Example: ` {{'description':'handle missing values in DataFrame'}} `

* **3. Combined API Path, Description, and Target Features Query (JSON Object):**
    Use this when you need to query a specific API, provide a description about its core functionality,parameter (to help retrieve information even if the exact API path has changed), and also specify the **exact features or aspects you are looking for** within its documentation.
    Example: ` {{'path': 'pandas.DataFrame.read_csv', 'description': 'Reads a CSV file into a DataFrame. Parameters include `filepath_or_buffer` (str, Path, or file-like object) and `sep` (str, default \',\'). Returns a new `DataFrame`.', 'target_features': ['new encoding options', 'chunking', 'date parsing']}} `

**Crucial Rule:** Only generate queries for APIs or functionalities that you are **uncertain about** or strongly **suspect require modification** due to the new dependency versions. Prioritize queries that address specific changes, deprecations, or potential incompatibilities. Do not generate queries for unchanged or clearly understood parts of the code. 

---

### Provided Information for Refactoring

**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

** Generated Target Code:**
{generated_target_code}

---

Now, generate the query list based on the instructions above.

### Generated Query List to retrieve context for code migration(if query contain API name, only public method,class,function,attribute,property for target_dependency {target_dependency} should be included, python builtin package should be excluded, like subprocess, os, time, etc.)
'''
# gemini optimized
VersiBCB_VACE_RAG_complete_withTargetCode_v1= '''
You are an expert Python code migration engineer. Your primary goal is to identify and generate precise queries for information on APIs and functionalities that require changes when Python dependencies are migrated.

You will be provided with:
1.  **A detailed description of the code's functionality.**
2.  **The original code snippet, including all its dependencies and their specific versions.**
3.  **The target dependencies with their new, upgraded versions.**
4.  **The generated target code.**

Your task is to generate a list of precise queries that will be used to retrieve relevant information from a RAG (Retrieval Augmented Generation) database for target code revision. These queries must specifically target APIs or functionalities where the usage, behavior, or availability might have changed, or that you are unfamiliar with, in the context of the `target_dependency` versions.

---
### Query Format Guidelines
For each query, use one of the following JSON formats. Ensure the output is a valid JSON list.

* **1. API Path Query (Single String):**
    Use this when you need information solely about an API's path.
    Example: ` {{'path':'pandas.DataFrame.read_csv'}} `
    Example: ` {{'path':'numpy.array.flatten'}} `

* **2. Functionality Description Query (Single String):**
    Use this when you need information solely about a sub-functionality.
    Example: ` {{'description':'calculate the mean of column in DataFrame'}} `
    Example: ` {{'description':'handle missing values in DataFrame'}} `

* **3. Combined API Path, Description, and Target Features Query (JSON Object):**
    Use this when you need to query a specific API, provide a description about its core functionality, parameter (to help retrieve information even if the exact API path has changed), and also specify the **exact features or aspects you are looking for** within its documentation.
    Example: ` {{'path': 'pandas.DataFrame.read_csv', 'description': 'Reads a CSV file into a DataFrame. Parameters include `filepath_or_buffer` (str, Path, or file-like object) and `sep` (str, default \',\'). Returns a new `DataFrame`.', 'target_features': ['new encoding options', 'chunking', 'date parsing']}} `

**Crucial Rule:** Only generate queries for APIs or functionalities that you are **uncertain about** or strongly **suspect require modification** due to the new dependency versions. Prioritize queries that address specific changes, deprecations, or potential incompatibilities. Do not generate queries for unchanged or clearly understood parts of the code.

---
### Provided Information for Refactoring
**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

**Generated Target Code:**
{generated_target_code}

---
Now, generate the query list based on the instructions above.
### Generated Query List to retrieve context for code migration (If a query contains an API name, only public methods, classes, functions, attributes, and properties for `target_dependency` {target_dependency} should be included. Python built-in packages like `subprocess`, `os`, `time`, etc., should be excluded.)
'''
VersiBCB_VACE_RAG_complete_withTargetCode_v2= '''
You are an expert Python code migration engineer. Your primary goal is to identify and generate precise queries for information on APIs and functionalities that require changes when Python dependencies are migrated.

You will be provided with:
1.  **A detailed description of the code's functionality.**
2.  **The original code snippet, including all its dependencies and their specific versions.**
3.  **The target dependencies with their new, upgraded versions.**
4.  **The generated target code.**

Your task is to generate a list of precise queries that will be used to retrieve relevant information from a RAG (Retrieval Augmented Generation) database for target code revision. These queries must specifically target APIs or functionalities where the usage, behavior, or availability might have changed, or that you are unfamiliar with, in the context of the `target_dependency` versions.

---
### Query Format Guidelines
For each query, use one of the following JSON formats. Ensure the output is a valid JSON list.

* **1. API Path Query (Single String):**
    Use this when you need information solely about an API's path.
    Example: ` {{'path':'pandas.DataFrame.read_csv'}} `
    Example: ` {{'path':'numpy.array.flatten'}} `

* **2. Functionality Description Query (Single String):**
    Use this when you need information solely about a sub-functionality.
    Example: ` {{'description':'calculate the mean of column in DataFrame'}} `
    Example: ` {{'description':'handle missing values in DataFrame'}} `

* **3. Combined API Path, Description, and Target Features Query (JSON Object):**
    Use this when you need to query a specific API, provide a description about its core functionality, parameter (to help retrieve information even if the exact API path has changed), and also specify the **exact features or aspects you are looking for** within its documentation.
    Example: ` {{'path': 'pandas.DataFrame.read_csv', 'description': 'Reads a CSV file into a DataFrame. Parameters include `filepath_or_buffer` (str, Path, or file-like object) and `sep` (str, default \',\'). Returns a new `DataFrame`.', 'target_features': ['new encoding options', 'chunking', 'date parsing']}} `

**Crucial Rule:** Only generate queries for APIs or functionalities that you are **uncertain about** or strongly **suspect require modification** due to the new dependency versions. Prioritize queries that address specific changes, deprecations, or potential incompatibilities. Do not generate queries for unchanged or clearly understood parts of the code.

---
### Provided Information for Refactoring
**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

**Generated Target Code:**
{generated_target_code}

---
Now, generate the query list based on the instructions above.
### Generated Query List to retrieve context for code migration (If a query contains an API name, only public methods, classes, functions, attributes, and properties for `target_dependency` {target_dependency} should be included. Python built-in packages like `subprocess`, `os`, `time`, etc., should be excluded.)
'''
VersiBCB_VACE_RAG_complete_withTargetCode_v3= '''
You are an expert Python code migration engineer. Your primary goal is to identify and generate precise queries for information on APIs and functionalities that require changes when Python dependencies are migrated.

You will be provided with:
1.  **A detailed description of the code's functionality.**
2.  **The original code snippet, including all its dependencies and their specific versions.**
3.  **The target dependencies with their new, upgraded versions.**
4.  **The generated target code.**

Your task is to generate a list of precise queries that will be used to retrieve relevant information from a RAG (Retrieval Augmented Generation) database for target code revision. These queries must specifically target APIs or functionalities where the usage, behavior, or availability might have changed, or that you are unfamiliar with, in the context of the `target_dependency` versions.

---
### Query Format Guidelines
For each query, use the following format. Ensure the output is a valid JSONL. Make sure queries to be separated by \\n only.
 Combined API Path, Description:**
    Use this when you need to query a specific API, provide a description about its core functionality, parameter (to help retrieve information even if the exact API path has changed)
    Example: ` {{'path': 'pandas.DataFrame.read_csv', 'description': 'Reads a CSV file into a DataFrame. Parameters include `filepath_or_buffer` (str, Path, or file-like object) and `sep` (str, default \',\'). Returns a new `DataFrame`.'}} `

**Crucial Rule:** Only generate queries for APIs or functionalities that you are **uncertain about** or strongly **suspect require modification** due to the new dependency versions. Prioritize queries that address specific changes, deprecations, or potential incompatibilities. Do not generate queries for unchanged or clearly understood parts of the code.

---
### Provided Information for Refactoring
**Functionality Description:**
{description}

**Original Dependencies and Versions:**
{origin_dependency}

**Original Code:**
{origin_code}

**Target Dependencies and New Versions:**
{target_dependency}

**Generated Target Code:**
{generated_target_code}

---
### examples

{{'path':'packA.moduleA.classA','description':'descriptionA'}}
{{'path':'packA.moduleB.classB','description':'descriptionB'}}
{{'path':'packB.moduleA.classA','description':'descriptionA'}}
{{'path':'packB.moduleB.classB','description':'descriptionB'}}
...




Now, generate the queries based on the instructions above.make sure queries are separated by \\n only.
### Generated Queries to retrieve context for code migration (If a query contains an API name, only public methods, classes, functions, attributes, and properties for `target_dependency` {target_dependency} should be included. Python built-in packages like `subprocess`, `os`, `time`, etc., should be excluded. )
'''
VersiBCB_VACE_RAG_complete_v4='''
You are an expert Python code migration engineer. Your goal is to generate highly precise and relevant search queries to find information needed for migrating Python code. You will be provided with details about the original code, its dependencies, the target dependencies, and a description of the code's functionality.

**Your task is to identify specific APIs or functionalities in the `Original Code` that are likely to require changes or verification when migrating to the `Target Dependencies`. For each such potential change, generate a concise and focused query.**

**Crucial Rules:**
1.  **Only generate queries for APIs or functionalities that you are *uncertain about* or *strongly suspect require modification* due to the new dependency versions.**
2.  **Prioritize queries that address specific changes, deprecations, or potential incompatibilities.**
3.  **DO NOT generate queries for unchanged or clearly understood parts of the code.**
4.  **DO NOT include any conversational text, explanations, or additional formatting beyond the specified JSON output.**

---
### Query Format Guidelines (Strict JSONL Output)
Generate one JSON object per line. The output MUST be a valid JSONL format.

* Each query object MUST contain a "query" key.
* Optionally, if the query directly relates to a specific API path, include a "target_api" key with the full API path (e.g., `pandas.DataFrame.read_csv`). This helps focus the search.

**Example Output Format:**
```json
{{"query": "What is the equivalent of the 'inplace' parameter for `pandas.DataFrame.fillna` in pandas of target dependency, which is used to fill NaN values directly in the DataFrame?", "target_api": "pandas.DataFrame.fillna"}}
{{"query": "How to achieve column-wise mean calculation for a DataFrame in pandas of target dependency?", "target_api": "pandas.DataFrame.mean"}}
{{"query": "What is the equivalent of `numpy.array.flatten` in numpy of target dependency, which is used to return a copy of the array collapsed into one dimension?", "target_api": "numpy.array.flatten"}}

```
in above, xxx can refer to parameter, function, class, attribute, property, etc. <functionality> refer to the functionality of the API. which should be fit in insteaf of use
---
### Provided Information for Migration:
Functionality Description:
{description}

Original Dependencies and Versions:
{origin_dependency}

Original Code:
{origin_code}

Target Dependencies and New Versions:
{target_dependency}
---
### Generated Queries (JSONL format, each query on a new line, make sure queries strongly related to the target functionality which is unknown for you):
'''

VersiBCB_VSCC_RAG_complete_v0= '''
You are an expert Python code generation assistant. Your goal is to generate highly precise and relevant search queries to find information needed for implementing a Python function. You will be provided with a description of the desired function's functionality and its target dependencies.

**Your task is to identify specific APIs or syntax within the target dependencies that are required to implement the described functionality, for which you are uncertain about the *exact correct usage or parameters*. For each such uncertain area, generate a concise and focused query to determine the precise API call.**

**Crucial Rules:**
1.  **Only generate queries for APIs or functionalities for which you are *uncertain about the correct syntax, required parameters, or direct method to achieve a specific sub-task* given the target dependencies.**
2.  **Prioritize queries that directly ask for the correct way to call a specific API or use a specific feature to achieve a part of the functionality, focusing solely on correctness.**
3.  **DO NOT generate queries for general Python syntax, common algorithms not directly involving the specific target libraries/versions, or clearly understood parts of the functionality that do not present an API usage challenge.**
4.  **DO NOT include any conversational text, explanations, or additional formatting beyond the specified JSON output.**

---
### Query Format Guidelines (Strict JSONL Output)
Generate one JSON object per line. The output MUST be a valid JSONL format.

* Each query object MUST contain a "query" key.
* Optionally, if the query directly relates to a specific API path, include a "target_api" key with the full API path (e.g., `pandas.DataFrame.read_csv`). This helps focus the search.

Below is an example of the functionality description and dependency:
Example Functionality Description:
"{{\"description\": [\"Create a Pandas DataFrame from a list of pairs and visualize the data using a bar chart.\", \"- The title of the barplot should be set to 'Category vs Value'`.\"], \"notes\": [], \"params\": [\"list_of_pairs (list of tuple): Each tuple contains:\", \"str: Category name.\", \"int: Associated value.\"], \"returns\": [\"tuple:\", \"DataFrame: A pandas DataFrame with columns 'Category' and 'Value'.\", \"Axes: A matplotlib Axes displaying a bar chart of categories vs. values.\"], \"reqs\": [\"pandas\", \"matplotlib.pyplot\", \"seaborn\"], \"raises\": [], \"examples\": [\">>> list_of_pairs = [('Fruits', 5), ('Vegetables', 9)]\", \">>> df, ax = task_func(list_of_pairs)\", \">>> print(df)\", \"Category  Value\", \"0      Fruits      5\", \"1  Vegetables      9\"]}}"
Dependency:
{{
            "matplotlib": "3.5.3",
            "pandas": "1.4.4",
            "python": "3.8",
            "seaborn": "0.13.2"
}}
**Example Output Format:**
```json
{{"query": "What is the correct syntax to create a `pandas.DataFrame` from a list of tuples with specified column names in pandas of target dependency?", "target_api": "pandas.DataFrame"}}
{{"query": "How to correctly set the title of a matplotlib plot using `matplotlib.pyplot` in target dependency?", "target_api": "matplotlib.pyplot.title"}}
{{"query": "What are the required parameters for `seaborn.barplot` when plotting two columns from a pandas DataFrame using seaborn of target dependency?", "target_api": "seaborn.barplot"}}
```

### Provided Information for Code Generation:

## Functionality Description: {description} Target Dependencies and New Versions: {target_dependency}

### Generated Queries (JSONL format, each query on a new line, make sure queries strongly related to the target functionality and necessary for its implementation based on the correct usage of target dependencies):

'''
VersiBCB_VSCC_RAG_complete_v1= '''
You are an expert Python code generation assistant. Your goal is to generate highly precise and relevant search queries to find information needed for implementing a Python function. You will be provided with a description of the desired function's functionality and its target dependencies.

**Your task is to identify specific APIs or syntax within the target dependencies that are required to implement the described functionality, for which you are uncertain about the *exact correct usage or parameters*. For each such uncertain area, generate a concise and focused query to determine the precise API call.**

**Crucial Rules:**
1.  **Only generate queries for APIs or functionalities for which you are *uncertain about the correct syntax, required parameters, or direct method to achieve a specific sub-task* given the target dependencies.**
2.  **Prioritize queries that directly ask for the correct way to call a specific API or use a specific feature to achieve a part of the functionality, focusing solely on correctness.**
3.  **DO NOT generate queries for general Python syntax, common algorithms not directly involving the specific target libraries/versions, or clearly understood parts of the functionality that do not present an API usage challenge.**
4.  **DO NOT include any conversational text, explanations, or additional formatting beyond the specified JSON output.**

---
### Query Format Guidelines (Strict JSONL Output)
Generate one JSON object per line. The output MUST be a valid JSONL format.

* Each query object MUST contain a "query" key.
* Optionally, if the query directly relates to a specific API path, include a "target_api" key with the full API path (e.g., `pandas.DataFrame.read_csv`). This helps focus the search.

**Example Output Format:**
```json
{{"query": "What is the equivalent of the 'inplace' parameter for `pandas.DataFrame.fillna` in pandas of target dependency, which is used to fill NaN values directly in the DataFrame?", "target_api": "pandas.DataFrame.fillna"}}
{{"query": "How to achieve column-wise mean calculation for a DataFrame in pandas of target dependency?", "target_api": "pandas.DataFrame.mean"}}
{{"query": "What is the equivalent of `numpy.array.flatten` in numpy of target dependency, which is used to return a copy of the array collapsed into one dimension?", "target_api": "numpy.array.flatten"}}
```

### Provided Information for Code Generation:

## Functionality Description: {description} Target Dependencies and New Versions: {target_dependency}

### Generated Queries (JSONL format, each query on a new line, make sure queries strongly related to the target functionality and necessary for its implementation based on the correct usage of target dependencies):

'''
VersiBCB_VSCC_RAG_complete_v2= '''
You are an expert Python code generation assistant. Your goal is to generate highly precise and relevant search queries to find information needed for implementing a Python function. You will be provided with a description of the desired function's functionality and its target dependencies.

**Your task is to identify specific APIs or syntax within the target dependencies that are required to implement the described functionality, for which you are uncertain about the *exact correct usage or parameters*. For each such uncertain area, generate a concise and focused query to determine the precise API call.**

**Crucial Rules:**
1.  **Only generate queries for APIs or functionalities for which you are *uncertain about the correct syntax, required parameters, or direct method to achieve a specific sub-task* given the target dependencies.**
2.  **Prioritize queries that directly ask for the correct way to call a specific API or use a specific feature to achieve a part of the functionality, focusing solely on correctness.**
3.  **DO NOT generate queries for general Python syntax, common algorithms not directly involving the specific target libraries/versions, or clearly understood parts of the functionality that do not present an API usage challenge.**
4.  **DO NOT include any conversational text, explanations, or additional formatting beyond the specified JSON output.**
---
### Query Format Guidelines (Strict JSONL Output)
Generate one JSON object per line. The output MUST be a valid JSONL format.

* Each query object MUST contain a "query" key.
* Each query object MUST include a "target_api" key with the full API path (e.g., `pandas.DataFrame.read_csv`) that is mostly likely to be related to the query. This helps focus the search.
* Each query object could include a "relevent_apis" key with a list of API paths that are likely to be related to the query. This helps focus the search.

**Example Output Format:**
```json
{{"query": "What is the equivalent of the 'inplace' parameter for `pandas.DataFrame.fillna` in pandas of target dependency, which is used to fill NaN values directly in the DataFrame?", "target_api": "pandas.DataFrame.fillna"}}
{{"query": "How to achieve column-wise mean calculation for a DataFrame in pandas of target dependency?", "target_api": "pandas.DataFrame.mean"}}
{{"query": "What is the equivalent of `numpy.array.flatten` in numpy of target dependency, which is used to return a copy of the array collapsed into one dimension?", "target_api": "numpy.array.flatten"}}
```

### Provided Information for Code Generation:

## Functionality Description: {description} Target Dependencies and New Versions: {target_dependency}

### Generated Queries (JSONL format, each query on a new line, make sure queries strongly related to the target functionality and necessary for its implementation based on the correct usage of target dependencies. Notably, do not generate queries that you are sure about the correct usage in target dependencies):

'''
VersiBCB_VSCC_REVIEW='''
Please generate queries that will be used to look up content in documentation of packages to solve errors for below code in target dependency.
functionality description:
{description}
target dependency:
{target_dependency}
generated code that try to achieve the functionality:
{code}
errors detected by static analysis:
{error_info} 
**Example Output Format:**
```json
{{"query": "what is the equivalent of `pandas.DataFrame.fillna` in pandas of target dependency, which is used to fill NaN values directly in the DataFrame?", "target_api": "pandas.DataFrame.fillna"}}
```
### generated queries(JSONL format, each query on a new line):
'''
VersiBCB_VACE_RAG_complete_v4_origin='''
You are an expert Python code migration engineer. Your goal is to generate highly precise and relevant search queries to find information needed for migrating Python code. You will be provided with details about the original code, its dependencies, the target dependencies, and a description of the code's functionality.

**Your task is to identify specific APIs or functionalities in the `Original Code` that are likely to require changes or verification when migrating to the `Target Dependencies`. For each such potential change, generate a concise and focused query.**

**Crucial Rules:**
1.  **Only generate queries for APIs or functionalities that you are *uncertain about* or *strongly suspect require modification* due to the new dependency versions.**
2.  **Prioritize queries that address specific changes, deprecations, or potential incompatibilities.**
3.  **DO NOT generate queries for unchanged or clearly understood parts of the code.**
4.  **DO NOT include any conversational text, explanations, or additional formatting beyond the specified JSON output.**

---
### Query Format Guidelines (Strict JSONL Output)
Generate one JSON object per line. The output MUST be a valid JSONL format.

* Each query object MUST contain a "query" key.
* Optionally, if the query specifically targets a known API, include a "target_api" key with the full API path (e.g., `pandas.DataFrame.read_csv`). This helps focus the search.

**Example Output Format:**
```json
{{"query": "changes to pandas.DataFrame.read_csv parameters in pandas, specifically for 'sep' and 'encoding'", "target_api": "pandas.DataFrame.read_csv"}}
{{"query": "how to handle missing values (NaN) in pandas.Series with pandas 2.1.0", "target_api": "pandas.Series.fillna"}}
{{"query": "what is the equivalent of numpy.array.flatten in numpy 1.26.0"}}
```
---
### Provided Information for Migration:
Functionality Description:
{description}

Original Dependencies and Versions:
{origin_dependency}

Original Code:
{origin_code}

Target Dependencies and New Versions:
{target_dependency}
---
### Generated Queries (JSONL format, each query on a new line):
'''
# 下面似乎太过丰富
VersiBCB_VACE_RAG_complete_v4_examples='''
You are an expert Python code migration engineer. Your goal is to generate highly precise and relevant search queries to find information needed for migrating Python code. You will be provided with details about the original code, its dependencies, the target dependencies, and a description of the code's functionality.

**Your task is to identify specific APIs or functionalities in the `Original Code` that are likely to require changes or verification when migrating to the `Target Dependencies`. For each such potential change, generate a concise and focused query.**

**Crucial Rules:**
1.  **Only generate queries for APIs or functionalities that you are *uncertain about* or *strongly suspect require modification* due to the new dependency versions.**
2.  **Prioritize queries that address specific changes, deprecations, or potential incompatibilities.**
3.  **DO NOT generate queries for unchanged or clearly understood parts of the code.**
4.  **DO NOT include any conversational text, explanations, or additional formatting beyond the specified JSON output.**

---
### Query Format Guidelines (Strict JSONL Output)
Generate one JSON object per line. The output MUST be a valid JSONL format.

* Each query object MUST contain a "query" key.
* Optionally, if the query specifically targets a known API, include a "target_api" key with the full API path (e.g., `pandas.DataFrame.read_csv`). This helps focus the search.
**Example Output Format:**
```json
{{"query": "What are the parameter changes for `pandas.DataFrame.read_csv` (reads a CSV file into a DataFrame, parameters include `filepath_or_buffer`, `sep`) in pandas 2.1.0, specifically regarding `encoding` options and `chunksize` for large files?", "target_api": "pandas.DataFrame.read_csv"}}
{{"query": "How to handle missing values (NaN) in `pandas.Series` using `fillna` (fills NA/NaN values) in pandas 2.1.0? Are there new default behaviors for `inplace` or recommended methods?", "target_api": "pandas.Series.fillna"}}
{{"query": "What is the recommended method or direct equivalent for `numpy.array.flatten` (returns a copy of the array collapsed into one dimension) in numpy 1.26.0, considering potential performance differences for large arrays?", "target_api": "numpy.array.flatten"}}
{{"query": "Are there any deprecations or changes to array creation functions like `numpy.zeros` (creates an array of zeros) in numpy 1.26.0?", "target_api": "numpy.zeros"}}
{{"query": "What are the changes to the behavior of element-wise operations between DataFrames (like addition or multiplication) in pandas 2.1.0, especially concerning index alignment and NaN propagation?"}}
```
---
### Provided Information for Migration:
Functionality Description:
{description}

Original Dependencies and Versions:
{origin_dependency}

Original Code:
{origin_code}

Target Dependencies and New Versions:
{target_dependency}
---
### Generated Queries (JSONL format, each query on a new line, make sure no duplicate queries):
'''
