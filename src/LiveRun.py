import argparse
from RAGv4 import RAGPipeline, key_from_file

from src.old.Reporter_old import Reporter
r = Reporter("../logs")

# Frame Config
FRAME_CONFIG = {
    "APIKey": key_from_file("key.txt"),
    "kb_source": "../data/DND.SRD.Wiki-0.5.2",
    "ind_dir": "../index",
}

# Setting Configs
BASIC_CONFIG = {
    "ind_name": "token-default",
    "rag_llm_model": "mistralai/mistral-7b-instruct-v0.1",
    "meta_llm_model": "openai/gpt-4o-mini",
}

BEST_CONFIG = {
    "ind_name": "sentences-markdown",
    "rag_llm_model": "mistralai/mistral-7b-instruct-v0.1",
    "meta_llm_model": "openai/gpt-4o-mini",
    "embed_split_on": "sentences",
    "embed_with_markdown": True,
    "chunk_size": 512,
    "chunk_overlay": 50,
    "retriever_top_k": 10,
    "retriever_query_variants": 1,
    "retriever_with_keywords": False,
    "use_llm_consolidation": True,    # the key module
    "use_dedup": False,
    "prompt_pretext": (
        "Use ONLY the context below to answer. "
        "If the context describes a condition or rule, you may reason about "
        "what it does and does not restrict. "
        "If the answer cannot be determined from the context at all, "
        "respond with: 'I cannot find this in the provided rules.' "
        "Do not use outside knowledge beyond what the context implies."
    ),
    "response_mode": "COMPACT",
}

# Session Names
SESSION_NAME = "LiveDemo"
CONFIG_PREFIX = "DEMO-"

NO_RAG_SUFFIX = "no-rag"
BASIC_SUFIX = "baseline"
BEST_SUFIX = "best-consolidation"

# Eval Configs
NO_EVAL_CONFIG = {
    "eval_faithfulness": False,
    "eval_relevancy": False,
    "eval_correctness": False,
    "eval_context_relevancy": False,
}

EVAL_CONFIG = {
    "eval_faithfulness": True,
    "eval_relevancy": True,
    "eval_correctness": True,
    "eval_context_relevancy": True,
}

def setup(mode, do_eval, do_verbose):
    frame_ = FRAME_CONFIG
    if do_eval:
        eval_ = EVAL_CONFIG
    else:
        eval_ = NO_EVAL_CONFIG

    session_name_ = SESSION_NAME
    if mode == "no_rag":
        config_name_ = CONFIG_PREFIX + NO_RAG_SUFFIX
        config_ = BASIC_CONFIG
    elif mode == "basic":
        config_name_ = CONFIG_PREFIX + BASIC_SUFIX
        config_ = BASIC_CONFIG
    elif mode == "best":
        config_name_ = CONFIG_PREFIX + BEST_SUFIX
        config_ = BEST_CONFIG
    else:
        raise ValueError(f"Unknown mode {mode}")

    RAG = RAGPipeline(**{**frame_, **eval_, **config_}, session_name=session_name_, config_name=config_name_, verbose=do_verbose)
    return RAG

def startup(RAG, mode):
    if mode == "no_rag":
        return
    config_path = RAG.get_config_path()
    print(config_path)
    r.terminal_print_config(config_path)

def cycle(RAG, mode):
    while True:
        prompt = input(f"{mode} > ")
        if prompt == "exit" or prompt == "quit" or prompt == "q":
            break
        if mode == "no_rag":
            RAG.simple_query(prompt)
        else:
            RAG.RAG_query(prompt)
        query_path = RAG.get_last_query_path()
        if query_path:
            r.terminal_print_query(query_path)



def main(args):
    mode = args.mode
    do_eval = args.eval
    do_verbose = args.verbose
    RAG = setup(mode, do_eval, do_verbose)
    startup(RAG, mode)
    cycle(RAG, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='LiveRun',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-m', '--mode', type=str, choices=['no_rag', 'basic', 'best'], default='basic')
    parser.add_argument("-e", "--eval", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
