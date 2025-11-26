"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines.
"""

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import t2ebm
import t2ebm.llm
from t2ebm.llm import AbstractChatModel

from t2ebm.graphs import extract_graph, graph_to_text

import t2ebm.prompts as prompts

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

###################################################################################################
# Talk to the EBM about other things than graphs.
###################################################################################################


def feature_importances_to_text(ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]):
    """Convert the feature importances of an EBM to text.

    Args:
        ebm (_type_): The EBM.

    Returns:
        str: Textual representation of the feature importances.
    """
    feature_importances = ""
    for feature_idx, feature_name in enumerate(ebm.feature_names_in_):
        feature_importances += (
            f"{feature_name}: {ebm.term_importances()[feature_idx]:.2f}\n"
        )
    return feature_importances


################################################################################################################
# Ask the LLM to perform high-level tasks directly on the EBM.
################################################################################################################


def describe_graph(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    feature_index: int,
    num_sentences: int = 7,
    **kwargs,
):
    """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

    The function accepts additional keyword arguments that are passed to extract_graph, graph_to_text, and describe_graph_cot.

    Args:
        llm (Union[AbstractChatModel, str]): The LLM.
        ebm (Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]): The EBM.
        feature_index (int): The index of the feature to describe.
        num_sentences (int, optional): The desired number of senteces for the description. Defaults to 7.

    Returns:
        str:  The description of the graph.
    """

    # llm setup
    llm = t2ebm.llm.setup(llm)

    # extract the graph from the EBM
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    graph = extract_graph(ebm, feature_index, **extract_dict)

    # convert the graph to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    # Pass ebm and feature_index for caching
    graph = graph_to_text(graph, ebm=ebm, feature_index=feature_index, **to_text_dict)

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {k: kwargs[k] for k in dict(kwargs) if k in llm_descripe_kwargs}
    messages = prompts.describe_graph_cot(
        graph, num_sentences=num_sentences, **llm_descripe_dict
    )

    # execute the prompt
    messages = t2ebm.llm.chat_completion(llm, messages)

    # the last message contains the summary
    return messages[-1]["content"]


def describe_ebm(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    num_sentences: int = 30,
    max_features: int = 5,  # Limit the number of features to analyze for performance
    **kwargs,
):
    """Ask the LLM to describe an EBM. 

    The function accepts additional keyword arguments that are passed to extract_graph, graph_to_text, and describe_graph_cot.

    Args:
        llm (Union[AbstractChatModel, str]): The LLM.
        ebm (Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]): The EBM.
        num_sentences (int, optional): The desired number of senteces for the description. Defaults to 30.
        max_features (int, optional): Maximum number of features to analyze. Defaults to 5 for performance.

    Returns:
        str: The description of the EBM.
    """

    # llm setup
    llm = t2ebm.llm.setup(llm)

    # Get feature importances to prioritize the most important features
    feature_importances = feature_importances_to_text(ebm)
    
    # Calculate feature importance scores
    importance_scores = ebm.term_importances()
    feature_indices = list(range(len(ebm.feature_names_in_)))
    
    # Sort features by absolute importance (most impactful first)
    sorted_indices = sorted(feature_indices, key=lambda i: abs(importance_scores[i]), reverse=True)
    
    # Take only the top N features for analysis (performance optimization)
    top_feature_indices = sorted_indices[:max_features]
    
    print(f"Analyzing top {len(top_feature_indices)} features out of {len(feature_indices)} total features")

    # Extract and process only the top features
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    
    graphs = []
    for feature_index in top_feature_indices:
        graphs.append(extract_graph(ebm, feature_index, **extract_dict))

    # convert the graphs to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    # Pass ebm and feature_index for caching
    graphs = [
        graph_to_text(graph, ebm=ebm, feature_index=feature_index, **to_text_dict) 
        for graph, feature_index in zip(graphs, top_feature_indices)
    ]

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {
        k: kwargs[k]
        for k in dict(kwargs)
        if k in llm_descripe_kwargs and k != "num_sentences"
    }
    messages = [
        prompts.describe_graph_cot(graph, num_sentences=5, **llm_descripe_dict)  # Reduced from 7 to 5
        for graph in graphs
    ]

    # Helper function for parallel processing
    def process_feature(args):
        """Process a single feature description."""
        idx, msg, feature_name = args
        print(f"Processing feature: {feature_name}")
        result = t2ebm.llm.chat_completion(llm, msg)[-1]["content"]
        return idx, result

    # Execute the prompts in parallel for better performance
    print(f"Processing {len(messages)} features in parallel...")
    graph_descriptions_dict = {}
    
    # Use ThreadPoolExecutor for parallel API calls
    # max_workers=3 to avoid rate limiting while still gaining speedup
    max_workers = min(3, len(messages))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for parallel processing
        tasks = [
            (i, msg, ebm.feature_names_in_[top_feature_indices[i]])
            for i, msg in enumerate(messages)
        ]
        
        # Submit all tasks
        futures = {executor.submit(process_feature, task): task[0] for task in tasks}
        
        # Collect results as they complete
        for future in as_completed(futures):
            idx, result = future.result()
            graph_descriptions_dict[idx] = result
            print(f"Completed feature {idx + 1}/{len(messages)}")
    
    # Reconstruct ordered list from dict
    graph_descriptions_list = [graph_descriptions_dict[i] for i in range(len(messages))]

    # combine the graph descriptions in a single string
    graph_descriptions = "\n\n".join(
        [
            ebm.feature_names_in_[top_feature_indices[idx]] + ": " + graph_description
            for idx, graph_description in enumerate(graph_descriptions_list)
        ]
    )

    print("Generating final summary...")

    # now, ask the llm to summarize the different descriptions
    llm_summarize_kwargs = list(inspect.signature(prompts.summarize_ebm).parameters)
    llm_summarize_dict = {
        k: kwargs[k] for k in dict(kwargs) if k in llm_summarize_kwargs
    }
    messages = prompts.summarize_ebm(
        feature_importances,
        graph_descriptions,
        num_sentences=num_sentences,
        **llm_summarize_dict,
    )

    # execute the prompt and return the summary
    return t2ebm.llm.chat_completion(llm, messages)[-1]["content"]
