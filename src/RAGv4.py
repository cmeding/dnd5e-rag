from llama_index.llms.openrouter import OpenRouter
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings,
    StorageContext, load_index_from_storage, PromptTemplate,
    get_response_synthesizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    AutoMergingRetriever,
    QueryFusionRetriever,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor,
    LongContextReorder,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    TransformQueryEngine,
    SubQuestionQueryEngine,
)
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    ContextRelevancyEvaluator,
)

import phoenix as px
import os
from loguru import logger
from pathlib import Path
import time
import json
from datetime import datetime


def key_from_file(path):
    with open(path) as f:
        return f.readline().strip()


class RAGPipeline:

    CATEGORY_KEYWORDS = [
        "Characterizations", "Classes", "Equipment", "Gamemastering",
        "Gameplay", "Monsters", "Races", "Spells", "Treasure",
    ]

    def __init__(
        self,
        APIKey=key_from_file("./key.txt"),
        kb_source="../data/DND.SRD.Wiki-0.5.2",
        ind_dir="../index",
        ind_name="SRD-BaseTest-v1",
        # LLMs
        rag_llm_model="mistralai/mistral-7b-instruct-v0.1",
        meta_llm_model="openai/gpt-4o-mini",
        meta_temperature=0.0,
        llm_max_response=512,
        llm_context_window=4096,
        llm_temperature=0.1,
        llm_top_p=0.9,
        llm_top_k=40,
        llm_instruction="",
        # Embedding
        embed_model="BAAI/bge-small-en-v1.5",
        embed_device="cpu",
        embed_context_window=4096,
        embed_budget=512,
        # Chunking
        chunk_size=512,
        chunk_overlay=50,
        embed_with_markdown=False,
        embed_split_on="tokens",
        special_mode="none",
        generate_metadata=True,
        # Retrieval
        retriever_top_k=10,
        retriever_query_variants=1,
        retriever_with_keywords=False,
        # Postprocessing
        post_use_cutoff=False,
        post_cutoff=0.14,
        post_use_rerank=False,
        post_rerank_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        post_rerank_top_n=3,
        post_use_reorder=False,
        # Prompt
        prompt_pretext=None,
        prompt_context_text="Context:",
        prompt_question_text="Question:",
        prompt_post_text="",
        prompt_answer_text="Answer:",
        response_mode="COMPACT",
        # Query transformations
        use_hyde=False,
        use_query_rewrite=False,
        use_query_decomposition=False,
        # Context processing
        use_dedup=False,
        dedup_threshold=0.85,
        use_llm_consolidation=False,
        # Guards
        use_confidence_guard=False,
        confidence_cutoff=0.5,
        # Cost tracking
        cost_per_1m_rag_prompt=0.11,
        cost_per_1m_rag_completion=0.19,
        cost_per_1m_meta_prompt=0.15,
        cost_per_1m_meta_completion=0.60,
        cost_per_1m_embed=0.00,
        # Evaluation
        eval_faithfulness=False,
        eval_relevancy=False,
        eval_context_relevancy=False,
        eval_correctness=False,
        eval_reference=None,
        # Setup
        do_startup_setup=True,
        start_phoenix=False,
        verbose=True,
        # Logging
        config_name="",
        session_name="",
        log_dir="../logs",
        log_queries=True,
    ):
        """
        RAG Pipeline for DnD 5e SRD question answering.

        Wraps LlamaIndex components into a configurable retrieval-augmented
        generation pipeline with built-in evaluation, cost tracking, and
        JSON session logging.

        Parameters
        ----------
        APIKey : str
            OpenRouter API key. Defaults to reading from ``./key.txt``.
        kb_source : str
            Path to the knowledge base directory containing ``.md`` source files.
        ind_dir : str
            Root directory where vector indices are persisted.
        ind_name : str
            Subfolder name for this index inside ``ind_dir``.
            **Change this value whenever any indexing parameter changes.**

        LLMs
        ----
        rag_llm_model : str
            OpenRouter model string for the RAG answer LLM.
        meta_llm_model : str
            OpenRouter model string for the meta LLM used in query rewrite,
            HyDE, decomposition, consolidation, and evaluation.
        meta_temperature : float
            Sampling temperature for the meta LLM. Keep at ``0.0`` for
            deterministic evaluation and transformation calls.
        llm_max_response : int
            Maximum number of tokens in any LLM response.
        llm_context_window : int
            LLM context window size in tokens. Affects how many chunks fit
            in a single prompt.
        llm_temperature : float
            Sampling temperature for the RAG answer LLM.
        llm_top_p : float
            Nucleus sampling probability for both LLMs.
        llm_top_k : int
            Top-k sampling for both LLMs.
        llm_instruction : str
            Additional instruction appended to the RAG LLM system prompt.

        Embedding
        ---------
        embed_model : str
            HuggingFace model name for chunk and query embedding.
            **Changing this requires rebuilding the index.**
        embed_device : str
            Device for embedding inference. ``"cpu"`` or ``"cuda"``.
        embed_context_window : int
            Passed to ``Settings.context_window``.
        embed_budget : int
            Passed to ``Settings.num_output`` (max LLM output tokens used
            when computing prompt budget).

        Chunking / Indexing
        -------------------
        All parameters in this section **require rebuilding the index**
        (i.e. a new ``ind_name``) when changed.

        chunk_size : int
            Maximum chunk size in tokens. Only used when
            ``special_mode="none"``.
        chunk_overlay : int
            Token overlap between consecutive chunks. Only used when
            ``special_mode="none"``.
        embed_with_markdown : bool
            If ``True``, prepends a ``MarkdownNodeParser`` before the text
            splitter. Valid for all ``special_mode`` values.
        embed_split_on : str
            Text splitter to use when ``special_mode="none"``.

            - ``"tokens"``    — ``TokenTextSplitter``
            - ``"sentences"`` — ``SentenceSplitter``

            Ignored when ``special_mode`` is not ``"none"``.
        special_mode : str
            Node parsing strategy. Overrides ``embed_split_on`` and
            ``chunk_size`` when not ``"none"``.

            - ``"none"``            — flat chunks via ``embed_split_on``
            - ``"sentence_window"`` — ``SentenceWindowNodeParser`` (window=3)
            - ``"hierarchical"``    — ``HierarchicalNodeParser`` ([2048,512,128])

        generate_metadata : bool
            If ``True``, attaches ``category`` and ``topic`` metadata to
            each node derived from the source file path.

        Retrieval
        ---------
        retriever_top_k : int
            Number of chunks to retrieve per query.
        retriever_query_variants : int
            Number of query variants to generate via ``QueryFusionRetriever``.
            ``1`` disables variant generation.
        retriever_with_keywords : bool
            If ``True``, adds a BM25 keyword retriever alongside the vector
            retriever and fuses results via reciprocal rerank.

        Postprocessing
        --------------
        post_use_cutoff : bool
            Filter out retrieved nodes below ``post_cutoff`` similarity score.
        post_cutoff : float
            Minimum similarity score threshold for ``SimilarityPostprocessor``.
        post_use_rerank : bool
            Re-rank retrieved nodes with a cross-encoder model.
        post_rerank_model : str
            HuggingFace cross-encoder model name for reranking.
        post_rerank_top_n : int
            Number of nodes to keep after reranking.
        post_use_reorder : bool
            Apply ``LongContextReorder`` to move highest-scoring nodes to
            the edges of the context window.

        Prompt
        ------
        prompt_pretext : str or None
            Instruction placed before the context block. ``None`` uses the
            default strict-grounding instruction.
        prompt_context_text : str
            Label for the context block. Default ``"Context:"``.
        prompt_question_text : str
            Label for the question. Default ``"Question:"``.
        prompt_post_text : str
            Optional text inserted between the question and the answer label.
        prompt_answer_text : str
            Answer prefix shown to the LLM. Default ``"Answer:"``.
        response_mode : str
            How retrieved chunks are combined for the LLM call.

            - ``"COMPACT"``          — stuff chunks into fewest calls (default)
            - ``"REFINE"``           — iterative refinement over each chunk
            - ``"SIMPLE_SUMMARIZE"`` — single call, truncates if needed
            - ``"TREE_SUMMARIZE"``   — builds a summary tree

        Query Transformations
        ---------------------
        ``use_hyde`` and ``use_query_decomposition`` are mutually exclusive.

        use_hyde : bool
            Generate a hypothetical answer document via the meta LLM and
            embed that for retrieval instead of the raw query (HyDE).
            Incompatible with ``use_dedup`` and ``use_llm_consolidation``.
        use_query_rewrite : bool
            Rewrite the query via the meta LLM before retrieval to make it
            more retrieval-friendly.
        use_query_decomposition : bool
            Decompose the query into sub-questions via the meta LLM and
            answer each independently before synthesising a final answer.

        Context Processing
        ------------------
        Incompatible with ``use_hyde``.

        use_dedup : bool
            Remove near-duplicate retrieved chunks before synthesis using
            Jaccard similarity.
        dedup_threshold : float
            Jaccard similarity threshold above which a chunk is considered
            a duplicate and removed.
        use_llm_consolidation : bool
            Merge all retrieved chunks into a single coherent summary via
            the meta LLM before passing to the answer LLM.

        Guards
        ------
        use_confidence_guard : bool
            If ``True``, refuse to answer when the best retrieval score is
            below ``confidence_cutoff``.
        confidence_cutoff : float
            Minimum best-node score required to proceed with answer
            generation when ``use_confidence_guard=True``.

        Cost Tracking
        -------------
        All prices in USD per 1 million tokens. Check OpenRouter for
        current model rates.

        cost_per_1m_rag_prompt : float
        cost_per_1m_rag_completion : float
        cost_per_1m_meta_prompt : float
        cost_per_1m_meta_completion : float
        cost_per_1m_embed : float

        Evaluation
        ----------
        All evaluators use the meta LLM.

        eval_faithfulness : bool
            Is the answer grounded in the retrieved context?
        eval_relevancy : bool
            Does the answer address the question that was asked?
        eval_correctness : bool
            Is the answer factually correct compared to ``eval_reference``?
        eval_context_relevancy : bool
            Were the retrieved chunks relevant to the question?
        eval_reference : str or None
            Ground-truth answer string required for correctness evaluation.

        Setup
        -----
        do_startup_setup : bool
            If ``False``, defers the ``_setup()`` call for manual
            initialisation.
        start_phoenix : bool
            Launch the Arize Phoenix tracing UI on setup.

        Observability
        -------------
        verbose : bool
            Print query info, retrieved nodes, token counts, costs, and
            evaluation results to stdout during each query.

        Logging
        -------
        config_name : str
            Human-readable label for this pipeline configuration, written
            to ``config.json`` and used in the session folder name.
        session_name : str
            Human-readable label for this run/session.
        log_dir : str
            Root directory for session log folders.
        log_queries : bool
            Write a ``query_NNN.json`` file for every query.
        """
        self.APIKey = APIKey
        self.kb_source = kb_source
        self.ind_dir = ind_dir
        self.ind_name = ind_name
        self.rag_llm_model = rag_llm_model
        self.meta_llm_model = meta_llm_model
        self.meta_temperature = meta_temperature
        self.embed_model = embed_model
        self.embed_device = embed_device
        self.llm_max_response = llm_max_response
        self.llm_context_window = llm_context_window
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.llm_top_k = llm_top_k
        self.llm_instruction = llm_instruction
        self.chunk_size = chunk_size
        self.chunk_overlay = chunk_overlay
        self.embed_context_window = embed_context_window
        self.embed_budget = embed_budget
        self.embed_with_markdown = embed_with_markdown
        self.embed_split_on = embed_split_on
        self.special_mode = special_mode
        self.generate_metadata = generate_metadata
        self.retriever_top_k = retriever_top_k
        self.retriever_query_variants = retriever_query_variants
        self.retriever_with_keywords = retriever_with_keywords
        self.post_use_cutoff = post_use_cutoff
        self.post_cutoff = post_cutoff
        self.post_use_rerank = post_use_rerank
        self.post_rerank_model = post_rerank_model
        self.post_rerank_top_n = post_rerank_top_n
        self.post_use_reorder = post_use_reorder
        self.prompt_pretext = prompt_pretext
        self.prompt_context_text = prompt_context_text
        self.prompt_question_text = prompt_question_text
        self.prompt_post_text = prompt_post_text
        self.prompt_answer_text = prompt_answer_text
        self.response_mode = response_mode
        self.use_hyde = use_hyde
        self.use_query_rewrite = use_query_rewrite
        self.use_query_decomposition = use_query_decomposition
        self.use_dedup = use_dedup
        self.dedup_threshold = dedup_threshold
        self.use_llm_consolidation = use_llm_consolidation
        self.use_confidence_guard = use_confidence_guard
        self.confidence_cutoff = confidence_cutoff
        self.cost_per_1m_rag_prompt = cost_per_1m_rag_prompt
        self.cost_per_1m_rag_completion = cost_per_1m_rag_completion
        self.cost_per_1m_meta_prompt = cost_per_1m_meta_prompt
        self.cost_per_1m_meta_completion = cost_per_1m_meta_completion
        self.cost_per_1m_embed = cost_per_1m_embed
        self.eval_faithfulness = eval_faithfulness
        self.eval_relevancy = eval_relevancy
        self.eval_context_relevancy = eval_context_relevancy
        self.eval_correctness = eval_correctness
        self.eval_reference = eval_reference
        self.start_phoenix = start_phoenix
        self.verbose = verbose
        self.config_name = config_name
        self.session_name = session_name
        self.log_dir = log_dir
        self.log_queries = log_queries
        self.query_engine = None

        self._default_pretext = (
            "Use ONLY the context below to answer. "
            "If the answer is not explicitly stated in the context, "
            "respond with exactly: 'I cannot find this in the provided rules.' "
            "Do not infer, guess, or use outside knowledge."
        )

        if do_startup_setup:
            self._setup()

    # ─────────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────────

    def _init_session(self, index, setup_time):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._session_id = f"{self.config_name}-{self.session_name}-{stamp}"
        self._session_dir = os.path.join(self.log_dir, self._session_id)
        self._query_count = 0
        os.makedirs(self._session_dir, exist_ok=True)

        config = {
            "session_id": self._session_id,
            "config_name": self.config_name,
            "session_name": self.session_name,
            "timestamp": datetime.now().isoformat(),
            "setup_time": setup_time,
            "pipeline": {k: getattr(self, k) for k in [
                "ind_name", "rag_llm_model", "meta_llm_model", "embed_model",
                "chunk_size", "chunk_overlay", "embed_split_on", "embed_with_markdown",
                "special_mode", "retriever_top_k", "retriever_query_variants",
                "retriever_with_keywords", "post_use_cutoff", "post_cutoff",
                "post_use_rerank", "post_rerank_model", "post_rerank_top_n",
                "post_use_reorder", "use_hyde", "use_query_rewrite",
                "use_query_decomposition", "use_dedup", "use_llm_consolidation",
                "use_confidence_guard", "confidence_cutoff", "response_mode",
                "eval_faithfulness", "eval_relevancy", "eval_correctness", "eval_context_relevancy",
            ]},
            "index": {
                "index_time": self.index_time,
                "index_mode": self.index_mode,
                "chunks_in_store": len(index.docstore.docs),
                "index_size_mb": self._get_index_size_mb(),
                "doc_count": self.doc_count,
            },
            "prompt": {
                "pretext": self.prompt_pretext or self._default_pretext,
                "context_text": self.prompt_context_text,
                "question_text": self.prompt_question_text,
                "post_text": self.prompt_post_text,
                "answer_text": self.prompt_answer_text,
                "response_mode": self.response_mode,
                "system_prompt": f"You are a DnD 5e rules expert. {self.llm_instruction}",
            },
        }

        with open(os.path.join(self._session_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Session {} created", self._session_id)

    def _save_query_record(self, query_record):
        if not self.log_queries:
            return
        fname = f"query_{self._query_count:03d}.json"
        with open(os.path.join(self._session_dir, fname), "w") as f:
            json.dump(query_record, f, indent=2)

    def _build_query_record(self, query, used_query, response, tokens, costs, timings, refused, confidence, eval_results):
        self._query_count += 1
        return {
            "query_id": f"query_{self._query_count:03d}",
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "query": {
                "original": query,
                "used":     used_query,
            },
            "retrieved_nodes": [
                {
                    "rank":     i + 1,
                    "score":    float(node.score),
                    "category": node.metadata.get("category"),
                    "topic":    node.metadata.get("topic"),
                    "file":     node.metadata.get("file_path"),
                    "preview":  node.text[:200],
                }
                for i, node in enumerate(getattr(response, "source_nodes", None) or [])
            ],
            "tokens": tokens,
            "costs": costs,
            "timings": timings,
            "answer": str(response),
            "refused": refused,
            "confidence": confidence,
            "evaluation": eval_results,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Logging Helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _get_index_size_mb(self) -> float:
        target_dir = os.path.join(self.ind_dir, self.ind_name)
        if not os.path.exists(target_dir):
            return 0.0
        total_bytes = sum(
            f.stat().st_size for f in Path(target_dir).rglob("*") if f.is_file()
        )
        return round(total_bytes / (1024 ** 2), 2)

    def _collect_eval_results(self, query: str, response) -> dict:
        results = {}

        if self.eval_faithfulness and self.evaluator_faithfulness:
            try:
                r = self.evaluator_faithfulness.evaluate_response(response=response)
                results["faithfulness"] = {
                    "score": r.score,
                    "passing": r.passing,
                    "feedback": r.feedback if r.feedback else None,
                }
            except Exception as e:
                results["faithfulness"] = {"error": str(e)}

        if self.eval_relevancy and self.evaluator_relevancy:
            try:
                r = self.evaluator_relevancy.evaluate_response(query=query, response=response)
                results["relevancy"] = {
                    "score": r.score,
                    "passing": r.passing,
                    "feedback": r.feedback if r.feedback else None,
                }
            except Exception as e:
                results["relevancy"] = {"error": str(e)}

        if self.eval_correctness and self.evaluator_correctness and self.eval_reference:
            try:
                r = self.evaluator_correctness.evaluate(
                    query=query, response=str(response), reference=self.eval_reference,
                )
                results["correctness"] = {
                    "score": r.score,
                    "passing": r.passing,
                    "feedback": r.feedback if r.feedback else None,
                }
            except Exception as e:
                results["correctness"] = {"error": str(e)}

        if self.eval_context_relevancy and self.evaluator_context_relevancy:
            try:
                r = self.evaluator_context_relevancy.evaluate_response(
                    query=query, response=response,
                )
                results["context_relevancy"] = {
                    "score": r.score,
                    "passing": r.passing,
                    "feedback": r.feedback if r.feedback else None,
                }
            except Exception as e:
                results["context_relevancy"] = {"error": str(e)}

        return results

    # ─────────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────────

    def _setup(self):
        t_total = time.time()
        logger.info("Setup RAG Pipeline")

        self.debug_handler = LlamaDebugHandler()
        Settings.callback_manager = CallbackManager([self.debug_handler])
        self._meta_prompt_tokens = 0
        self._meta_completion_tokens = 0

        logger.info("Setup LLMs")
        t = time.time()
        Settings.llm = self.rag_llm_config = self._configure_llm(
            APIKey=self.APIKey,
            model=self.rag_llm_model,
            max_tokens=self.llm_max_response,
            context_window=self.llm_context_window,
            system_instruction=self.llm_instruction,
            temperature=self.llm_temperature,
            top_p=self.llm_top_p,
            top_k=self.llm_top_k,
        )
        self.meta_llm_config = self._configure_llm(
            APIKey=self.APIKey,
            model=self.meta_llm_model,
            max_tokens=self.llm_max_response,
            context_window=self.llm_context_window,
            temperature=self.meta_temperature,
            top_p=self.llm_top_p,
            top_k=self.llm_top_k,
        )
        logger.info("LLMs ready [{}]", self._elapsed(t))

        logger.info("Setup Embedding Model")
        t = time.time()
        Settings.embed_model = self._configure_embed(
            model=self.embed_model,
            device=self.embed_device,
        )
        logger.info("Embedding Model ready [{}]", self._elapsed(t))

        logger.info("Checking for existing Vector Index...")
        t = time.time()
        transformations = self._configure_parser(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlay,
            context_window=self.embed_context_window,
            answer_budget=self.embed_budget,
            use_markdown=self.embed_with_markdown,
            split=self.embed_split_on,
            special_mode=self.special_mode,
        )
        self.index_ref = index = self._index(
            kb_src=self.kb_source,
            ind_dir=self.ind_dir,
            ind_name=self.ind_name,
            transformations=transformations,
            use_metadata=self.generate_metadata,
        )
        logger.info("Index ready [{}]", self._elapsed(t))

        logger.info("Setup Query Engine")
        t = time.time()
        retriever = self._configure_retriever(
            index=index,
            top_k=self.retriever_top_k,
            queries=self.retriever_query_variants,
            use_keywords=self.retriever_with_keywords,
            special_mode=self.special_mode,
        )
        postprocessors = self._configure_postprocessor(
            special_mode=self.special_mode,
            use_cutoff=self.post_use_cutoff,
            cutoff=self.post_cutoff,
            use_rerank=self.post_use_rerank,
            rerank_model=self.post_rerank_model,
            rerank_top_n=self.post_rerank_top_n,
            use_reorder=self.post_use_reorder,
        )
        prompt_template = self._configure_prompt_template(
            pretext=self.prompt_pretext,
            context_text=self.prompt_context_text,
            question_text=self.prompt_question_text,
            post_text=self.prompt_post_text,
            answer_text=self.prompt_answer_text,
        )
        synthesizer = self._configure_response_synthesizer(
            prompt_template=prompt_template,
            response_mode=self.response_mode,
        )
        self.query_engine = self._configure_query_engine(
            retriever=retriever,
            synthesizer=synthesizer,
            postprocessor=postprocessors,
        )
        self._synthesizer = synthesizer
        logger.info("Query Engine ready [{}]", self._elapsed(t))

        self._setup_evaluators()

        if self.start_phoenix:
            t = time.time()
            logger.info("Setup Arize Phoenix")
            self._start_arize_phoenix()
            logger.info("Arize Phoenix ready [{}]", self._elapsed(t))

        setup_time = self._elapsed(t_total)
        logger.info("Pipeline ready [total: {}]", setup_time)
        self._init_session(index=index, setup_time=setup_time)

    def _setup_evaluators(self):
        self.evaluator_faithfulness = (
            FaithfulnessEvaluator(llm=self.meta_llm_config) if self.eval_faithfulness else None
        )
        self.evaluator_relevancy = (
            RelevancyEvaluator(llm=self.meta_llm_config) if self.eval_relevancy else None
        )
        self.evaluator_correctness = (
            CorrectnessEvaluator(llm=self.meta_llm_config) if self.eval_correctness else None
        )
        self.evaluator_context_relevancy = (  # ← add
            ContextRelevancyEvaluator(llm=self.meta_llm_config) if self.eval_context_relevancy else None
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────────────

    def _configure_llm(
        self, APIKey=None, model="mistralai/mistral-7b-instruct-v0.1",
        max_tokens=512, context_window=4096, system_instruction="",
        temperature=0.1, top_p=0.9, top_k=40,
    ):
        return OpenRouter(
            api_key=APIKey,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["###", "Question:", "Context:"],
            timeout=60.0,
            max_retries=3,
            context_window=context_window,
            system_prompt=f"You are a DnD 5e rules expert. Answer all Questions correctly. {system_instruction}",
            extra_body={
                "transforms": [],
                "route": "fallback",
            },
        )

    def _configure_embed(self, model="BAAI/bge-small-en-v1.5", device="cpu"):
        return HuggingFaceEmbedding(
            model_name=model,
            embed_batch_size=32,
            max_length=512,
            device=device,
            normalize=True,
            query_instruction="Represent this question for retrieval: ",
            text_instruction="Represent this text for retrieval: ",
        )

    def _configure_parser(
        self, chunk_size=512, chunk_overlap=50, context_window=4096,
        answer_budget=512, use_markdown=False, split="tokens", special_mode="none",
    ):
        Settings.num_output = answer_budget
        Settings.context_window = context_window

        if special_mode == "sentence_window":
            return [SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )]

        if special_mode == "hierarchical":
            return [HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128],
            )]

        parsers = []
        if use_markdown:
            parsers.append(MarkdownNodeParser())

        if split == "tokens":
            parsers.append(TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ",
            ))
        elif split == "sentences":
            parsers.append(SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
            ))
        else:
            logger.error("Unknown split type: {}", split)
            raise ValueError(f"Unknown split type: {split}")

        return parsers

    def _configure_retriever(
        self, index, top_k=10, queries=1, use_keywords=False, special_mode="none",
    ):
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        retrievers = [vector_retriever]

        if use_keywords:
            retrievers.append(BM25Retriever.from_defaults(
                nodes=index.docstore.docs.values(),
                similarity_top_k=top_k,
            ))

        fused = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=top_k,
            num_queries=queries,
            use_async=True,
            mode="reciprocal_rerank",
        )

        if special_mode == "hierarchical":
            return AutoMergingRetriever(
                vector_retriever,
                storage_context=index.storage_context,
                verbose=True,
            )

        return fused

    def _configure_postprocessor(
        self, special_mode="none", use_cutoff=False, cutoff=0.5,
        use_rerank=False, rerank_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        rerank_top_n=3, use_reorder=False,
    ):
        postprocessors = []

        if special_mode == "sentence_window":
            postprocessors.append(MetadataReplacementPostProcessor(
                target_metadata_key="window",
            ))
        elif special_mode not in ("none", "hierarchical"):
            logger.error("Unknown special mode: {}", special_mode)
            raise ValueError(f"Unknown special mode: {special_mode}")

        if use_cutoff:
            postprocessors.append(SimilarityPostprocessor(similarity_cutoff=cutoff))

        if use_rerank:
            postprocessors.append(SentenceTransformerRerank(
                model=rerank_model, top_n=rerank_top_n,
            ))

        if use_reorder:
            postprocessors.append(LongContextReorder())

        return postprocessors

    def _configure_response_synthesizer(self, prompt_template, response_mode="COMPACT"):
        mode_map = {
            "COMPACT":          ResponseMode.COMPACT,
            "REFINE":           ResponseMode.REFINE,
            "SIMPLE_SUMMARIZE": ResponseMode.SIMPLE_SUMMARIZE,
            "TREE_SUMMARIZE":   ResponseMode.TREE_SUMMARIZE,
        }
        mode = mode_map.get(response_mode, ResponseMode.COMPACT)
        return get_response_synthesizer(
            response_mode=mode,
            streaming=False,
            verbose=True,
            text_qa_template=prompt_template,
        )

    def _configure_query_engine(self, retriever, synthesizer, postprocessor):
        base_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessor,
        )

        if self.use_hyde:
            return TransformQueryEngine(
                query_engine=base_engine,
                query_transform=HyDEQueryTransform(
                    llm=self.meta_llm_config,
                    include_original=True,
                ),
            )

        if self.use_query_decomposition:
            return SubQuestionQueryEngine.from_defaults(
                query_engine_tools=[QueryEngineTool.from_defaults(
                    query_engine=base_engine,
                    name="dnd_srd",
                    description=(
                        "Contains all DnD 5e SRD rules including spells, classes, "
                        "conditions, combat mechanics, equipment, races and monsters."
                    ),
                )],
                question_gen=LLMQuestionGenerator.from_defaults(llm=self.meta_llm_config),
                response_synthesizer=synthesizer,
                use_async=True,
                verbose=True,
            )

        return base_engine

    def _configure_prompt_template(
        self, pretext=None, context_text="Context:", question_text="Question:",
        post_text="", answer_text="Answer:",
    ):
        if pretext is None:
            pretext = self._default_pretext
        template_string = (
            f"{pretext}\n\n"
            f"{context_text}\n{{context_str}}\n\n"
            f"{question_text} {{query_str}}\n"
            f"{f'{post_text}{chr(10)}' if post_text else ''}"
            f"{answer_text}"
        )
        return PromptTemplate(template_string)

    # ─────────────────────────────────────────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────────────────────────────────────────

    def _add_metadata(self, documents):
        for doc in documents:
            path = Path(doc.metadata["file_path"])
            doc.metadata["category"] = path.parts[-2]
            doc.metadata["topic"] = path.stem
        return documents

    def _index(self, kb_src, ind_dir, ind_name, transformations, use_metadata=True):
        target_dir = os.path.join(ind_dir, ind_name)
        t = time.time()
        if os.path.exists(target_dir):
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(target_dir))

            )
            self._print_index_info(index, was_loaded=True, index_dir=target_dir)
            self.index_mode = "loaded"
            self.doc_count = None
            self.index_time = self._elapsed(t)
            return index

        documents = SimpleDirectoryReader(
            input_dir=kb_src, recursive=True, required_exts=[".md"],
        ).load_data()
        self.doc_count = doc_count = len(documents)
        logger.info("Loaded {} documents from {}", doc_count, kb_src)

        if use_metadata:
            documents = self._add_metadata(documents)

        index = VectorStoreIndex.from_documents(
            documents, transformations=transformations, show_progress=True,
        )
        index.storage_context.persist(str(target_dir))
        self._print_index_info(index, was_loaded=False, doc_count=doc_count, index_dir=target_dir)
        self.index_mode = "generated"
        self.index_time = self._elapsed(t)
        return index

    # ─────────────────────────────────────────────────────────────────────────────
    # Phoenix
    # ─────────────────────────────────────────────────────────────────────────────

    def _start_arize_phoenix(self):
        px.launch_app()
        tracer_provider = register(
            project_name="RAGTester",
            endpoint="http://localhost:6006/v1/traces",
        )
        instrumentor = LlamaIndexInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(tracer_provider=tracer_provider)

    # ─────────────────────────────────────────────────────────────────────────────
    # Query helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        prompt = PromptTemplate(
            "Rewrite the following question to be more specific and retrieval-friendly "
            "for a DnD 5e rules database. Keep it concise. Return only the rewritten question.\n\n"
            "Original: {query}\n"
            "Rewritten:"
        )
        result = self.meta_llm_config.complete(prompt=prompt.format(query=query))
        self._extract_meta_usage(result)
        rewritten = result.text.strip()
        logger.info("Query rewrite: '{}' --> '{}'", query, rewritten)
        return rewritten

    def _deduplicate_nodes(self, nodes):
        unique, seen = [], []
        for node in nodes:
            words = set(node.text.lower().split())
            is_dup = any(
                len(words & set(s.lower().split())) /
                max(len(words | set(s.lower().split())), 1) > self.dedup_threshold
                for s in seen
            )
            if not is_dup:
                seen.append(node.text)
                unique.append(node)
            else:
                logger.info("Dedup removed: '{}'", node.text[:60])
        logger.info("Dedup: {} → {} nodes", len(nodes), len(unique))
        return unique

    def _consolidate_context(self, query: str, nodes) -> str:
        chunks = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]:\n{node.text}" for i, node in enumerate(nodes)
        )
        prompt = PromptTemplate(
            "Below are several DnD 5e rule excerpts retrieved for a question.\n"
            "They may overlap, be redundant, or contain tangential information.\n"
            "Consolidate them into a single coherent, deduplicated context summary.\n"
            "Keep all information relevant to the question. Remove redundancy.\n\n"
            "Question: {query}\n\n"
            "Excerpts:\n{chunks}\n\n"
            "Consolidated Context:"
        )
        result = self.meta_llm_config.complete(prompt.format(query=query, chunks=chunks))
        self._extract_meta_usage(result)
        consolidated = result.text.strip()
        logger.info("Consolidated: {} chunks → 1 summary", len(nodes))
        return consolidated

    def _check_retrieval_confidence(self, response) -> bool:
        if "sub_qa" in (response.metadata or {}):
            return True
        if not response.source_nodes:
            return False
        scores = [n.score for n in response.source_nodes if n.score is not None]
        if not scores:
            return True
        best = max(scores)
        if best < self.confidence_cutoff:
            logger.info("Low retrieval confidence: best score {:.4f}", best)
            return False
        return True

    def _extract_meta_usage(self, result):
        if not (hasattr(result, "raw") and result.raw):
            return
        raw = result.raw
        try:
            usage = raw.usage
            self._meta_prompt_tokens     += getattr(usage, "prompt_tokens", 0) or 0
            self._meta_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        except AttributeError:
            usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
            self._meta_prompt_tokens     += usage.get("prompt_tokens", 0)
            self._meta_completion_tokens += usage.get("completion_tokens", 0)

    def _calculate_costs(
        self, rag_prompt, rag_completion, meta_prompt, meta_completion, embed_tokens=0,
    ):
        rag_cost   = (rag_prompt   / 1_000_000) * self.cost_per_1m_rag_prompt \
                   + (rag_completion / 1_000_000) * self.cost_per_1m_rag_completion
        meta_cost  = (meta_prompt  / 1_000_000) * self.cost_per_1m_meta_prompt \
                   + (meta_completion / 1_000_000) * self.cost_per_1m_meta_completion
        embed_cost = (embed_tokens / 1_000_000) * self.cost_per_1m_embed
        return {
            "rag": rag_cost, "meta": meta_cost,
            "embed": embed_cost, "total": rag_cost + meta_cost + embed_cost,
        }

    def _elapsed(self, start: float) -> str:
        return f"{time.time() - start:.2f}s"

    # ─────────────────────────────────────────────────────────────────────────────
    # Print / observability
    # ─────────────────────────────────────────────────────────────────────────────

    def _print_index_info(self, index, was_loaded, doc_count=None, index_dir=None):
        if not self.verbose:
            return
        status = "Loaded from storage" if was_loaded else "Built from documents"
        docstore_count = len(index.docstore.docs)
        disk_size_str = "N/A"
        if index_dir and os.path.exists(index_dir):
            total_bytes = sum(
                f.stat().st_size for f in Path(index_dir).rglob("*") if f.is_file()
            )
            if   total_bytes < 1024:        disk_size_str = f"{total_bytes} B"
            elif total_bytes < 1024 ** 2:   disk_size_str = f"{total_bytes / 1024:.1f} KB"
            elif total_bytes < 1024 ** 3:   disk_size_str = f"{total_bytes / (1024 ** 2):.1f} MB"
            else:                           disk_size_str = f"{total_bytes / (1024 ** 3):.2f} GB"

        print(f"\n{'=' * 60}")
        print(f"  INDEX INFO")
        print(f"  Status:          {status}")
        if doc_count is not None:
            print(f"  Source docs:     {doc_count}")
        print(f"  Chunks in store: {docstore_count}")
        print(f"  Index size:      {disk_size_str}")
        print(f"{'=' * 60}\n")

    def _print_query_info(self, original, rewritten=None, category=None):
        if not self.verbose:
            return
        print(f"\n{'=' * 60}")
        print(f"  QUERY INFO")
        print(f"  Original:   {original}")
        if rewritten and rewritten != original:
            print(f"  Rewritten:  {rewritten}")
        if category:
            print(f"  Category:   {category}")
        print(f"{'=' * 60}\n")

    def _print_subquestions(self, response):
        if not self.verbose:
            return
        sub_qa = (response.metadata or {}).get("sub_qa")
        if not sub_qa:
            return
        print(f"\n{'=' * 60}")
        print(f"  SUB-QUESTIONS ({len(sub_qa)} generated)")
        for i, item in enumerate(sub_qa):
            sq = item.get("sub_q")
            ans = item.get("answer", "")
            q_text = sq.sub_question.query_str if sq else "?"
            print(f"\n  [{i + 1}] Q: {q_text}")
            print(f"       A: {str(ans)[:200]}{'...' if len(str(ans)) > 200 else ''}")
        print(f"{'=' * 60}\n")

    def _print_retrieved_nodes(self, response):
        if not self.verbose:
            return
        nodes = response.source_nodes
        if not nodes:
            print("\n  [RETRIEVAL] No nodes retrieved.\n")
            return
        print(f"\n{'=' * 60}")
        print(f"  RETRIEVED NODES ({len(nodes)} total)")
        for i, node in enumerate(nodes):
            score   = f"{float(node.score):.4f}" if node.score is not None else "N/A"
            file    = node.metadata.get("file_path", node.metadata.get("file_name", "unknown"))
            cat     = node.metadata.get("category", "?")
            topic   = node.metadata.get("topic", "?")
            preview = node.text[:200].replace("\n", " ")
            print(f"\n  Rank {i + 1} | Score: {score} | {cat}/{topic}")
            print(f"  File:    {file}")
            print(f"  Preview: {preview}{'...' if len(node.text) > 200 else ''}")
        print(f"{'=' * 60}\n")

    def _print_hyde_prompt(self):
        if not self.verbose or not self.use_hyde:
            return
        events = self.debug_handler.get_event_pairs(CBEventType.LLM)
        if not events:
            return
        print(f"\n{'=' * 60}")
        print(f"  HYDE — HYPOTHETICAL DOCUMENT")
        for i, (start, end) in enumerate(events):
            if not (hasattr(start, "payload") and start.payload):
                continue
            for msg in start.payload.get("messages", []):
                content = str(getattr(msg, "content", ""))
                if "Please write a passage" in content or "passage" in content.lower():
                    print(f"\n  Input prompt (Call #{i + 1}):")
                    print(f"  {content[:300]}")
                    if hasattr(end, "payload") and end.payload:
                        resp = end.payload.get("response", None)
                        if resp:
                            text = getattr(resp, "text", str(resp))
                            print(f"\n  Generated hypothetical document:")
                            print(f"  {text[:600]}{'...' if len(text) > 600 else ''}")
                    break
        print(f"{'=' * 60}\n")

    def _collect_llm_tokens(self) -> dict:
        events = self.debug_handler.get_llm_inputs_outputs()
        total_prompt = total_completion = total_calls = 0
        for event_pair in events:
            output = event_pair[1] if len(event_pair) > 1 else None
            prompt_tokens = completion_tokens = 0
            if output and hasattr(output, "raw") and output.raw:
                usage = output.raw.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            if prompt_tokens == 0 and output:
                prompt_tokens = len(str(event_pair[0])) // 4
                completion_tokens = len(str(output)) // 4
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_calls += 1

        rag_prompt = max(0, total_prompt - self._meta_prompt_tokens)
        rag_completion = max(0, total_completion - self._meta_completion_tokens)
        return {
            "rag_prompt": rag_prompt,
            "rag_completion": rag_completion,
            "meta_prompt": self._meta_prompt_tokens,
            "meta_completion": self._meta_completion_tokens,
            "total_calls": total_calls,
            "total": total_prompt + total_completion,
        }

    def _print_llm_usage(self, token_data: dict, t_query=None, t_total=None):
        # always reset counters and flush regardless of verbose
        self._meta_prompt_tokens = 0
        self._meta_completion_tokens = 0
        self.debug_handler.flush_event_logs()

        if not self.verbose:
            return

        if not token_data or token_data["total_calls"] == 0:
            print("\n  [LLM USAGE] No LLM events captured.\n")
            return

        rag_tokens = token_data["rag_prompt"] + token_data["rag_completion"]
        meta_tokens = token_data["meta_prompt"] + token_data["meta_completion"]
        costs = self._calculate_costs(
            token_data["rag_prompt"], token_data["rag_completion"],
            token_data["meta_prompt"], token_data["meta_completion"],
        )

        print(f"\n{'=' * 60}")
        print(f"  LLM CALLS ({token_data['total_calls']} total)")
        print(f"\n  ── TOTALS ──────────────────────────────")
        print(f"  Total calls:       {token_data['total_calls']}")
        print(f"  Total tokens:      {token_data['total']}")
        print(f"  RAG tokens:        {rag_tokens}")
        print(f"  Meta tokens:       {meta_tokens}")
        if t_query: print(f"  Query time:        {t_query}")
        if t_total: print(f"  Total time:        {t_total}")
        print(f"{'=' * 60}\n")

        self._print_costs(costs)

    def _print_costs(self, costs: dict):
        if not self.verbose:
            return
        print(f"\n{'=' * 60}")
        print(f"  ESTIMATED COSTS (USD)")
        print(f"  RAG LLM:    ${costs['rag']:.6f}   ({self.rag_llm_model})")
        print(f"  Meta LLM:   ${costs['meta']:.6f}   ({self.meta_llm_model})")
        if costs['embed'] > 0:
            print(f"  Embedding:  ${costs['embed']:.6f}   ({self.embed_model})")
        print(f"  ─────────────────────────────────────")
        print(f"  Total:      ${costs['total']:.6f}")
        print(f"{'=' * 60}\n")

    def _print_answer(self, response, refused=False):
        print(f"\n{'=' * 60}")
        print(f"  ANSWER")
        if refused:
            print("  [REFUSED] Retrieval confidence too low.")
        else:
            print(f"  {str(response)}")
        print(f"{'=' * 60}\n")

    def _run_evaluators(self, eval_results: dict):
        if not eval_results:
            return
        print(f"\n{'=' * 60}")
        print(f"  EVALUATION")
        for name, result in eval_results.items():
            if "error" in result:
                print(f"\n  {name.capitalize()}: [ERROR] {result['error']}")
            else:
                print(f"\n  {name.capitalize()}")
                print(f"    Score:    {result['score']}")
                print(f"    Passing:  {result['passing']}")
                print(f"    Feedback: {result.get('feedback', 'N/A')}")
        print(f"{'=' * 60}\n")



    # ─────────────────────────────────────────────────────────────────────────────
    # Public query interface
    # ─────────────────────────────────────────────────────────────────────────────

    def simple_query(self, query: str):
        self._meta_prompt_tokens = self._meta_completion_tokens = 0
        t_total = time.time()
        logger.info("Simple (No-RAG) query: {}", query)
        self.debug_handler.flush_event_logs()

        self._print_query_info(original=query)

        t = time.time()
        response = Settings.llm.complete(query)
        t_query = self._elapsed(t)
        logger.info("LLM query [{}]", t_query)

        print(f"\n{'=' * 60}")
        print(f"  ANSWER (No RAG — direct LLM)")
        print(f"  {str(response)}")
        print(f"{'=' * 60}\n")

        token_data = self._collect_llm_tokens()
        self._print_llm_usage(token_data=token_data, t_query=t_query, t_total=self._elapsed(t_total))

        eval_results = self._collect_eval_results(query=query, response=response)
        self._run_evaluators(eval_results=eval_results)

        tokens = token_data or {
            "rag_prompt": 0, "rag_completion": 0,
            "meta_prompt": 0, "meta_completion": 0,
            "total_calls": 0, "total": 0,
        }
        costs = self._calculate_costs(
            tokens["rag_prompt"], tokens["rag_completion"],
            tokens["meta_prompt"], tokens["meta_completion"],
        )
        timings = {
            "rag_s": None,
            "query_s": float(t_query.rstrip("s")),
            "total_s": float(self._elapsed(t_total).rstrip("s")),
        }

        record = self._build_query_record(
            query=query,
            used_query=query,
            response=response,
            tokens=tokens,
            costs=costs,
            timings=timings,
            refused=False,
            confidence={"passed": None, "best_score": None},
            eval_results=eval_results,
        )
        self._save_query_record(record)

        logger.info("Finished simple query [total: {}]", self._elapsed(t_total))


    def RAG_query(self, query: str):
        self._meta_prompt_tokens = self._meta_completion_tokens = 0
        t_total = time.time()
        logger.info("RAG query: {}", query)
        self.debug_handler.flush_event_logs()

        used_query = query

        if self.use_query_rewrite:
            t = time.time()
            used_query = self._rewrite_query(used_query)
            logger.info("Query rewrite [{}]", self._elapsed(t))

        self._print_query_info(
            original=query,
            rewritten=used_query if self.use_query_rewrite else None,
        )

        t = time.time()
        use_manual = (self.use_dedup or self.use_llm_consolidation) and not self.use_hyde

        if use_manual:
            from llama_index.core.schema import NodeWithScore, TextNode
            nodes = self.query_engine.retriever.retrieve(used_query)
            if self.use_dedup:
                nodes = self._deduplicate_nodes(nodes)
            if self.use_llm_consolidation:
                consolidated = self._consolidate_context(used_query, nodes)
                nodes = [NodeWithScore(node=TextNode(text=consolidated), score=1.0)]
            t_rag = self._elapsed(t_total)
            #response = self.query_engine.response_synthesizer.synthesize(
            #    query=used_query, nodes=nodes,
            #)
            response = self._synthesizer.synthesize(
                query=used_query,
                nodes=nodes,
            )
        else:
            if self.use_hyde and (self.use_dedup or self.use_llm_consolidation):
                logger.warning("dedup/consolidation incompatible with hyde — skipping")
            t_rag = self._elapsed(t_total)
            response = self.query_engine.query(used_query)

        t_query = self._elapsed(t)
        logger.info("Query [{}]", t_query)

        self._print_hyde_prompt()
        self._print_subquestions(response)
        self._print_retrieved_nodes(response)
        token_data = self._collect_llm_tokens()
        self._print_llm_usage(token_data=token_data, t_query=t_query, t_total=self._elapsed(t_total))

        confidence = self._check_retrieval_confidence(response)
        logger.info("Retrieval confidence: {}", "good" if confidence else "low")

        refused = self.use_confidence_guard and not confidence
        self._print_answer(response, refused=refused)
        logger.info("Finished RAG query [total: {}]", self._elapsed(t_total))

        eval_results = self._collect_eval_results(query=used_query, response=response)
        self._run_evaluators(eval_results=eval_results)
        logger.info("Finished Evaluation")

        logger.info(f"Save Query as JSON")

        tokens = token_data or {
            "rag_prompt": 0, "rag_completion": 0,
            "meta_prompt": 0, "meta_completion": 0,
            "total_calls": 0, "total": 0,
        }
        costs = self._calculate_costs(
            tokens["rag_prompt"], tokens["rag_completion"],
            tokens["meta_prompt"], tokens["meta_completion"],
        )
        timings = {
            "rag_s": float(t_rag.rstrip("s")),
            "query_s": float(t_query.rstrip("s")),
            "total_s": float(self._elapsed(t_total).rstrip("s")),
        }
        scores = [n.score for n in response.source_nodes if n.score is not None]
        confidence_info = {"passed": confidence,
                           "best_score": float(max(scores)) if scores else None
                           }

        record = self._build_query_record(
            query,
            used_query,
            response,
            tokens,
            costs,
            timings,
            refused,
            confidence_info,
            eval_results=eval_results,
        )
        self._save_query_record(record)

    def batch_simple_query(self, queries: list[str], references: list[str] | None = None) -> list[dict]:
        """
        Run simple_query (no RAG) for a list of questions.

        Parameters
        ----------
        queries : list[str]
            List of question strings to run.
        references : list[str] or None
            Optional list of ground-truth answer strings for correctness
            evaluation. Must be the same length as queries if provided.
            Overrides self.eval_reference for each individual query.

        Returns
        -------
        list[dict]
            List of query record dicts in the same order as input.
            Records are also saved to JSON if log_queries=True.
        """
        if references is not None and len(references) != len(queries):
            raise ValueError(
                f"references length ({len(references)}) must match queries length ({len(queries)})"
            )

        records = []
        total = len(queries)
        original_reference = self.eval_reference

        for i, query in enumerate(queries):
            if references is not None:
                self.eval_reference = references[i]

            logger.info("Batch simple query [{}/{}]: {}", i + 1, total, query)
            self.simple_query(query)

            record_path = os.path.join(
                self._session_dir, f"query_{self._query_count:03d}.json"
            )
            if os.path.exists(record_path):
                with open(record_path) as f:
                    records.append(json.load(f))

        self.eval_reference = original_reference
        logger.info("Batch simple complete — {} queries", total)
        return records

    def batch_RAG_query(self, queries: list[str], references: list[str] | None = None) -> list[dict]:
        """
        Run RAG_query for a list of questions.

        Parameters
        ----------
        queries : list[str]
            List of question strings to run.
        references : list[str] or None
            Optional list of ground-truth answer strings for correctness
            evaluation. Must be the same length as queries if provided.
            Overrides self.eval_reference for each individual query.

        Returns
        -------
        list[dict]
            List of query record dicts in the same order as input.
            Records are also saved to JSON if log_queries=True.
        """
        if references is not None and len(references) != len(queries):
            raise ValueError(
                f"references length ({len(references)}) must match queries length ({len(queries)})"
            )

        records = []
        total = len(queries)
        original_reference = self.eval_reference

        for i, query in enumerate(queries):
            # swap in per-query reference if provided
            if references is not None:
                self.eval_reference = references[i]

            logger.info("Batch RAG query [{}/{}]: {}", i + 1, total, query)
            self.RAG_query(query)

            record_path = os.path.join(
                self._session_dir, f"query_{self._query_count:03d}.json"
            )
            if os.path.exists(record_path):
                with open(record_path) as f:
                    records.append(json.load(f))

        # restore original reference
        self.eval_reference = original_reference
        logger.info("Batch RAG complete — {} queries", total)
        return records

def main():
    key = key_from_file("./key.txt")
    RAG = RAGPipeline(APIKey=key)
    while True:
        RAG.RAG_query(input("Input Query: > "))


if __name__ == "__main__":
    main()