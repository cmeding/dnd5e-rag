import argparse
from RAGv4 import RAGPipeline, key_from_file

from Reporter import Reporter
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
    "embed_split_on": "tokens",
    "embed_with_markdown": False,
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

PROMPT_STRICT = (
    "Use ONLY the context below to answer. "
    "If the answer is not explicitly stated in the context, "
    "respond with exactly: 'I cannot find this in the provided rules.' "
    "Do not infer, guess, or use outside knowledge."
)

PROMPT_SOFT = (
    "Answer the question using the context below as your primary source. "
    "You may use reasoning and inference based on the rules described. "
    "If the context contains no relevant information at all, say so."
)

STRICT_CONFIG = {
    "ind_name": "sentences-markdown",
    "prompt_pretext": PROMPT_STRICT,
    "post_use_rerank": True,
    "post_rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "retriever_top_k": 15,
    "post_rerank_top_n": 5,
    "use_query_decomposition": True,
    "use_query_rewrite": False,
    "use_hyde": False,
}

SOFT_CONFIG = {
    "ind_name": "sentences-markdown",
    "prompt_pretext": PROMPT_SOFT,
    "post_use_rerank": True,
    "post_rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "retriever_top_k": 15,
    "post_rerank_top_n": 5,
    "use_query_decomposition": True,
    "use_query_rewrite": False,
    "use_hyde": False,
}

SOFT_SIMPLE_CONFIG = {
    "ind_name": "sentences-markdown",
    "prompt_pretext": PROMPT_SOFT,
    "post_use_rerank": True,
    "post_rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "retriever_top_k": 15,
    "post_rerank_top_n": 5,
    "use_query_decomposition": False,
    "use_query_rewrite": False,
    "use_hyde": False,
}

BAD_CONFIG = {
    "ind_name": "token-default",        # worst chunking — no markdown, token splitting
    "prompt_pretext": PROMPT_STRICT,    # strict prompt — fails on Type B questions
    "retriever_top_k": 15,              # lots of noise in context
    "retriever_query_variants": 3,      # more fusion = more RRF score compression
    "response_mode": "REFINE",          # iterative refinement — worst response mode
}

# Session Names
SESSION_NAME = "LiveDemo"
CONFIG_PREFIX = "DEMO-"

NO_RAG_SUFFIX = "no-rag"
BASIC_SUFIX = "baseline"
BEST_SUFIX = "best-consolidation"
SOFT_SUFIX = "soft-prompt"
STRICT_SUFIX = "strict-prompt"
SOFT_SIMPLE_SUFFIX = "soft-simple"
BAD_SUFIX = "bad"

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
    if mode == "no-rag":
        config_name_ = CONFIG_PREFIX + NO_RAG_SUFFIX
        config_ = BASIC_CONFIG
    elif mode == "basic":
        config_name_ = CONFIG_PREFIX + BASIC_SUFIX
        config_ = BASIC_CONFIG
    elif mode == "best":
        config_name_ = CONFIG_PREFIX + BEST_SUFIX
        config_ = BEST_CONFIG
    elif mode == "soft":
        config_name_ = CONFIG_PREFIX + SOFT_SUFIX
        config_ = SOFT_CONFIG
    elif mode == "strict":
        config_name_ = CONFIG_PREFIX + STRICT_SUFIX
        config_ = STRICT_CONFIG
    elif mode == "soft-simple":
        config_name_ = CONFIG_PREFIX + SOFT_SIMPLE_SUFFIX
        config_ = SOFT_SIMPLE_CONFIG
    elif mode == "bad":
        config_name_ = CONFIG_PREFIX + BAD_SUFIX
        config_ = BAD_CONFIG
    else:
        raise ValueError(f"Unknown mode {mode}")

    RAG = RAGPipeline(**{**frame_, **eval_, **config_}, session_name=session_name_, config_name=config_name_, verbose=do_verbose)
    return RAG

def startup(RAG, mode):
    if mode == "no-rag":
        return
    config_path = RAG.get_config_path()
    print(config_path)
    r.terminal_print_config(config_path)

def cycle(RAG, mode):
    while True:
        prompt = input(f"{mode} > ")
        if prompt == "exit" or prompt == "quit" or prompt == "q":
            break
        if mode == "no-rag":
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

    parser.add_argument('-m', '--mode', type=str, choices=['no-rag', 'basic', 'best', 'soft', 'strict', 'soft-simple', 'bad'], default='basic')
    parser.add_argument("-e", "--eval", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
