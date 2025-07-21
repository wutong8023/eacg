import logging
import re
from benchmark.config.code.dataset2prompt import dataset2prompt
from utils.RAGutils.document_utils import get_version

def format_prompt(data, context, dataset, task, ban_deprecation=False, tokenizer=None, max_prompt_token_length=None, truncate=True, review_mode=False, generated_target_code=None, error_fix_mode=False, error_info=None, retrieved_info=""):
    """
    根据数据集和任务类型格式化prompt.
    如果提供了tokenizer和max_prompt_token_length, 会尝试截断context以确保整个prompt不超过指定长度.
    Args:
        data: dict, 输入数据
        context: str, 上下文内容
        dataset: str, 数据集名称 ('VersiCode' 或 'VersiBCB')
        task: str, 任务类型 ('VSCC' 或 'VACE')
        ban_deprecation: bool, 是否禁止废弃信息
        tokenizer: transformers.PreTrainedTokenizer, 用于计算token长度和截断
        max_prompt_token_length: int, prompt的最大允许token长度
        truncate: bool, 是否启用截断功能，默认为True,可以用于保证整个input在模型最大max_position_embeddings范围内
        review_mode: bool, 是否为review模式，用于代码review
        generated_target_code: str, 生成的目标代码，在review模式下使用
        error_fix_mode: bool, 是否为error_fix模式，用于修复代码错误
        error_info: str, 错误信息，在error_fix模式下使用
        retrieved_info: str, 检索到的信息，用于补充上下文
    Returns:
        str: 格式化后的prompt
    """
    BD_suffix = "_BD" if ban_deprecation else ""
    
    # Determine the prompt template based on mode
    if error_fix_mode and error_info:
        # 根据是否有retrieved_info选择不同的error_fix模板
        if retrieved_info:
            # 使用带检索信息的error_fix模板
            template_key = f"{task}{BD_suffix}_ERRORFIX_RETRIEVE"
            if template_key in dataset2prompt[dataset]:
                prompt_template = dataset2prompt[dataset][template_key]
            else:
                # 如果没有带检索的模板，回退到普通error_fix模板
                prompt_template = dataset2prompt[dataset][f"{task}{BD_suffix}_ERRORFIX"]
        else:
            # 使用普通error_fix模板
            prompt_template = dataset2prompt[dataset][f"{task}{BD_suffix}_ERRORFIX"]
        context_to_use = ""  # Error fix模式不使用RAG context
    elif review_mode:
        prompt_template = dataset2prompt[dataset][f"{task}{BD_suffix}_REVIEW"]
    else:
        prompt_template = dataset2prompt[dataset][f"{task}{BD_suffix}"]

    original_context = str(context) if context is not None else "" # Ensure context is a string
    
    # For error_fix_mode, skip context processing since it doesn't use RAG context
    if error_fix_mode and error_info:
        context_to_use = ""
    else:
        context_to_use = original_context

    # 只有在启用截断且提供了必要参数时才进行截断处理，且不在error_fix模式下
    if truncate and tokenizer and max_prompt_token_length is not None and max_prompt_token_length > 0 and not (error_fix_mode and error_info):
        # Phase 1: Determine shell prompt length and available space for context
        placeholder_keys = set(re.findall(r"\{(\w+)\}", prompt_template))
        shell_measurement_args = {key: "" for key in placeholder_keys} # Initialize all with empty strings

        # Populate shell_measurement_args with actual values for non-knowledge_doc fields
        # This needs to mirror the final formatting logic's keys.
        temp_data_for_shell = dict(data) # operate on a copy

        if dataset == 'VersiCode':
            shell_measurement_args.update({
                "description": temp_data_for_shell.get("description", ""),
                "dependency": temp_data_for_shell.get("dependency", ""),
            })
            if task == "VSCC":
                shell_measurement_args["version"] = get_version(temp_data_for_shell.get("version")) if temp_data_for_shell.get("version") is not None else ""
            elif task == "VACE":
                shell_measurement_args.update({
                    "origin_version": get_version(temp_data_for_shell.get("origin_version")) if temp_data_for_shell.get("origin_version") is not None else "",
                    "origin_code": temp_data_for_shell.get("origin_code", ""),
                    "target_version": get_version(temp_data_for_shell.get("target_version")) if temp_data_for_shell.get("target_version") is not None else ""
                })
        elif dataset == 'VersiBCB':
            shell_measurement_args.update({
                "description": temp_data_for_shell.get("description", ""),
            })
            if task == "VSCC":
                shell_measurement_args["dependency"] = temp_data_for_shell.get("dependency", "")
            elif task == "VACE":
                shell_measurement_args.update({
                    "origin_dependency": temp_data_for_shell.get("origin_dependency", ""),
                    "origin_code": temp_data_for_shell.get("origin_code", ""),
                    "target_dependency": temp_data_for_shell.get("target_dependency", "")
                })
        
        shell_measurement_args["knowledge_doc"] = "" # Ensure knowledge_doc is empty for shell measurement

        try:
            prompt_shell_str = prompt_template.format(**shell_measurement_args)
            # 使用tokenizer.encode计算总token长度（包括特殊token）
            shell_token_ids = tokenizer.encode(prompt_shell_str, add_special_tokens=True)
            shell_token_length = len(shell_token_ids)
            
            available_for_context_tokens = max_prompt_token_length - shell_token_length
            
            if available_for_context_tokens < 0:
                logging.warning(
                    f"Base prompt shell ({shell_token_length} tokens) already exceeds max_prompt_token_length ({max_prompt_token_length}). "
                    f"Context will be empty. Dataset: {dataset}, Task: {task}"
                )
                context_to_use = ""
            elif original_context: # If there was context to begin with and space is non-negative
                # 计算context的token长度（不包括特殊token，因为它是片段）
                context_token_ids = tokenizer.encode(original_context, add_special_tokens=False)
                
                if len(context_token_ids) > available_for_context_tokens:
                    truncated_context_token_ids = context_token_ids[:available_for_context_tokens]
                    # Decode without adding special tokens, and skip them if any were part of the encoding
                    context_to_use = tokenizer.decode(truncated_context_token_ids, skip_special_tokens=True)
                    logging.info(
                        f"Context truncated from {len(context_token_ids)} to {len(truncated_context_token_ids)} tokens "
                        f"to fit into available space of {available_for_context_tokens} tokens."
                    )
                # else: context_to_use remains original_context (it fits)
            else: # No original context, or no space and no original context
                context_to_use = "" # Ensure context is empty
                
            # 验证最终prompt的总token长度
            final_prompt_for_verification = prompt_template.format(**{**shell_measurement_args, "knowledge_doc": context_to_use})
            final_token_ids = tokenizer.encode(final_prompt_for_verification, add_special_tokens=True)
            final_token_length = len(final_token_ids)
            
            if final_token_length > max_prompt_token_length:
                logging.warning(
                    f"Final prompt ({final_token_length} tokens) still exceeds max_prompt_token_length ({max_prompt_token_length}). "
                    f"Expected: {shell_token_length + len(tokenizer.encode(context_to_use, add_special_tokens=False))}"
                )
            else:
                logging.debug(
                    f"Final prompt token length: {final_token_length}/{max_prompt_token_length} tokens. "
                    f"Shell: {shell_token_length}, Context: {len(tokenizer.encode(context_to_use, add_special_tokens=False))}"
                )
                
        except KeyError as e:
            logging.error(f"KeyError during shell prompt string calculation: {e}. Args: {shell_measurement_args}. Template: '{prompt_template}'. Context will not be truncated.")
            # context_to_use remains original_context if shell calculation fails
        except Exception as e_gen:
            logging.error(f"Generic error during shell prompt string calculation or context truncation: {e_gen}. Context will not be truncated.")
    elif not truncate:
        logging.debug("Truncation disabled, using original context as-is.")
    else:
        logging.debug("Truncation not performed: missing tokenizer or max_prompt_token_length.")

    # Phase 2: Prepare final arguments for the prompt template, using context_to_use
    final_args_dict = {}
    # Populate final_args_dict similar to shell_measurement_args, but with actual 'context_to_use'.
    # And ensure all data from 'data' dict is correctly referenced.
    if dataset == 'VersiCode':
        final_args_dict = {
            "knowledge_doc": context_to_use,
            "description": data.get("description",""),
            "dependency": data.get("dependency",""),
        }
        if task == "VSCC":
            final_args_dict["version"] = get_version(data.get("version")) if data.get("version") is not None else ""
        elif task == "VACE":
            final_args_dict.update({
                "origin_version": get_version(data.get("origin_version")) if data.get("origin_version") is not None else "",
                "origin_code": data.get("origin_code",""),
                "target_version": get_version(data.get("target_version")) if data.get("target_version") is not None else ""
            })
            # Add generated_target_code for review mode
            if review_mode and generated_target_code is not None:
                final_args_dict["generated_target_code"] = generated_target_code
            # Add error_info for error_fix mode
            if error_fix_mode and error_info is not None:
                final_args_dict["error_info"] = error_info
                if generated_target_code is not None:
                    final_args_dict["generated_target_code"] = generated_target_code
                # Add retrieved_info if available
                if retrieved_info:
                    final_args_dict["retrieved_info"] = retrieved_info
    elif dataset == 'VersiBCB':
        final_args_dict = {
            "knowledge_doc": context_to_use,
            "description": data.get("description",""),
        }
        if task == "VSCC":
            final_args_dict["dependency"] = data.get("dependency","")
        elif task == "VACE":
            final_args_dict.update({
                "origin_dependency": data.get("origin_dependency",""),
                "origin_code": data.get("origin_code",""),
                "target_dependency": data.get("target_dependency","")
            })
            # Add error_info for error_fix mode
            if error_fix_mode and error_info is not None:
                final_args_dict["error_info"] = error_info
                if generated_target_code is not None:
                    final_args_dict["generated_target_code"] = generated_target_code
                # Add retrieved_info if available
                if retrieved_info:
                    final_args_dict["retrieved_info"] = retrieved_info
        # Add generated_target_code for review mode
        if review_mode and generated_target_code is not None:
            final_args_dict["generated_target_code"] = generated_target_code
        # 如果error_info存在，则加入;error_info不局限于error_fix_mode
        if error_info:
            final_args_dict["error_info"] = error_info
    else: # Should have been caught by dataset2prompt access earlier
        raise ValueError(f"Invalid dataset: {dataset}")

    # Ensure all keys required by the template are in final_args_dict, defaulting to empty string if not found.
    # This prevents KeyErrors if a template has a placeholder not explicitly handled above for a given data item.
    all_template_keys = set(re.findall(r"\{(\w+)\}", prompt_template))
    for key in all_template_keys:
        if key not in final_args_dict:
            final_args_dict[key] = data.get(key, "") # Default to data.get(key, "") for safety

    try:
        final_prompt = prompt_template.format(**final_args_dict)
        
        # 如果启用了截断且提供了tokenizer，记录最终prompt的token统计信息
        if truncate and tokenizer:
            final_token_count = len(tokenizer.encode(final_prompt, add_special_tokens=True))
            logging.debug(f"Final formatted prompt token count: {final_token_count}")
            
        return final_prompt
    except KeyError as e:
        logging.error(f"KeyError during final prompt formatting: {e}. Args: {final_args_dict}. Template: '{prompt_template}'")
        # Fallback: return a simple representation or re-raise
        # For now, re-raise as it indicates a mismatch that should be fixed.
        raise 