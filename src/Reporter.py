import json
import os
import glob
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import textwrap

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # foreground
    WHITE   = "\033[97m"
    GREY    = "\033[37m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    # background
    BG_DARK = "\033[40m"

def bold(s):    return f"{C.BOLD}{s}{C.RESET}"
def dim(s):     return f"{C.DIM}{s}{C.RESET}"
def cyan(s):    return f"{C.CYAN}{s}{C.RESET}"
def green(s):   return f"{C.GREEN}{s}{C.RESET}"
def yellow(s):  return f"{C.YELLOW}{s}{C.RESET}"
def red(s):     return f"{C.RED}{s}{C.RESET}"
def blue(s):    return f"{C.BLUE}{s}{C.RESET}"
def magenta(s): return f"{C.MAGENTA}{s}{C.RESET}"
def grey(s):    return f"{C.DIM}{C.GREY}{s}{C.RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

W = 72   # line width

def _bar(char="─", width=W):
    return char * width

def _header(title, char="═"):
    pad = max(0, W - len(title) - 4)
    left = pad // 2
    right = pad - left
    return f"{char * left}  {bold(title)}  {char * right}"

def _score_color(score):
    if score is None:   return grey("N/A")
    if score >= 0.8:    return green(f"{score:.2f}")
    if score >= 0.5:    return yellow(f"{score:.2f}")
    return red(f"{score:.2f}")

def _passing(passing):
    if passing is True:  return green("✓ pass")
    if passing is False: return red("✗ fail")
    return grey("—")

def _trunc(text, n=80):
    if text is None: return grey("—")
    t = str(text).replace("\n", " ").strip()
    return t[:n] + dim("…") if len(t) > n else t


# ─────────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────────

class Reporter:

    def __init__(self, log_dir: str = "../logs"):
        self.log_dir = log_dir

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_session(self, session_id: str):
        path = os.path.join(self.log_dir, session_id)
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {path}")
        with open(config_path) as f:
            config = json.load(f)
        queries = []
        for qf in sorted(glob.glob(os.path.join(path, "query_*.json"))):
            with open(qf) as f:
                queries.append(json.load(f))
        return config, queries

    def list_sessions(self):
        sessions = []
        for d in sorted(glob.glob(os.path.join(self.log_dir, "*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
                sessions.append(os.path.basename(d))
        return sessions

    def load_all_sessions(self):
        result = []
        for sid in self.list_sessions():
            try:
                config, queries = self.load_session(sid)
                result.append({"config": config, "queries": queries})
            except Exception as e:
                print(red(f"  Could not load {sid}: {e}"))
        return result

    # ── Display: single query ────────────────────────────────────────────────

    def display_query(self, query: dict, show_nodes: bool = True,
                      show_tokens: bool = True, show_eval: bool = True):
        print()
        print(cyan(_header(f"QUERY  {query.get('query_id', '')}".strip())))
        q = query.get("query", {})
        print(f"  {bold('Original:')}  {q.get('original', '—')}")
        if q.get("used") and q.get("used") != q.get("original"):
            rewritten = cyan(q.get("used"))
            print(f"  {bold('Rewritten:')} {rewritten}")
        print(f"  {bold('Time:')}      {dim(query.get('timestamp', '—'))}")

        # answer
        print()
        print(bold("  ANSWER"))
        refused = query.get("refused", False)
        if refused:
            print(f"  {red('[REFUSED — low retrieval confidence]')}")
        else:
            answer = str(query.get("answer", "")).strip()
            for line in answer.split("\n"):
                print(f"  {line}")

        # retrieved nodes
        if show_nodes:
            nodes = query.get("retrieved_nodes", [])
            if nodes:
                print()
                print(bold(f"  RETRIEVED NODES  ({len(nodes)})"))
                print(f"  {grey(_bar('─', W - 2))}")
                for n in nodes:
                    score_s = _score_color(n.get("score"))
                    cat = n.get("category") or "?"
                    topic = n.get("topic") or "?"
                    rank = n.get("rank", "?")
                    rank_s = bold(f"#{rank}")
                    cat_s = cyan(cat)
                    preview = dim(_trunc(n.get("preview"), 90))
                    print(f"  {rank_s}  {score_s}  {cat_s}/{topic}")
                    print(f"      {preview}")

        # tokens & costs
        if show_tokens:
            tokens = query.get("tokens", {})
            costs = query.get("costs", {})
            timings = query.get("timings", {})
            if tokens:
                print()
                print(bold("  TOKENS & COST"))
                total_calls = tokens.get("total_calls", "—")
                total_tok = yellow(str(tokens.get("total", "—")))
                rag_tok = tokens.get("rag_prompt", 0) + tokens.get("rag_completion", 0)
                meta_tok = tokens.get("meta_prompt", 0) + tokens.get("meta_completion", 0)
                print(f"  Total calls:  {total_calls}")
                print(f"  Total tokens: {total_tok}")
                print(f"  RAG:          {rag_tok}")
                print(f"  Meta:         {meta_tok}")
                if costs:
                    cost_val = costs.get("total", 0)
                    cost_s = green(f"${cost_val:.6f}")
                    print(f"  Cost (USD):   {cost_s}")
                if timings:
                    qs = timings.get("query_s") or timings.get("total_s")
                    ts = timings.get("total_s")
                    if qs:
                        print(f"  Query time:   {qs}s")
                    if ts and ts != qs:
                        print(f"  Total time:   {ts}s")

        # evaluation
        if show_eval:
            evals = query.get("evaluation", {})
            if evals:
                print()
                print(bold("  EVALUATION"))
                for name, result in evals.items():
                    name_s = name.capitalize()
                    if "error" in result:
                        err_s = red("[ERROR]")
                        print(f"  {name_s:14s}  {err_s} {result['error']}")
                    else:
                        score = result.get("score")
                        passing = result.get("passing")
                        fb = _trunc(result.get("feedback"), 70)
                        score_s = _score_color(score)
                        passing_s = _passing(passing)
                        fb_s = dim(fb)
                        print(f"  {name_s:14s}  {score_s}  {passing_s}  {fb_s}")

        print(grey(_bar()))

    def display_session(self, session_id: str, show_queries: bool = True,
                        show_nodes: bool = False, show_tokens: bool = True,
                        show_eval: bool = True):
        config, queries = self.load_session(session_id)
        pipe = config.get("pipeline", {})

        print()
        print(magenta(_header(f"SESSION  {config.get('config_name', '')}  /  {config.get('session_name', '')}")))
        print(f"  {bold('ID:')}          {dim(config.get('session_id', ''))}")
        print(f"  {bold('Timestamp:')}   {config.get('timestamp', '—')}")
        print(f"  {bold('Setup time:')}  {config.get('setup_time', '—')}")

        # index
        idx = config.get("index", {})
        print()
        print(bold("  INDEX"))
        print(f"  Mode:    {cyan(idx.get('index_mode', '—'))}")
        print(f"  Chunks:  {idx.get('chunks_in_store', '—')}")
        print(f"  Size:    {idx.get('index_size_mb', '—')} MB")
        if idx.get("doc_count"):
            print(f"  Docs:    {idx['doc_count']}")

        # pipeline config
        print()
        print(bold("  PIPELINE CONFIG"))
        flags = [
            ("LLM",        pipe.get("rag_llm_model", "—")),
            ("Meta LLM",   pipe.get("meta_llm_model", "—")),
            ("Embed",      pipe.get("embed_model", "—")),
            ("Chunking",   f"{pipe.get('embed_split_on','—')}  chunk={pipe.get('chunk_size','—')}  overlap={pipe.get('chunk_overlay','—')}"),
            ("Mode",       pipe.get("special_mode", "none")),
            ("Top-k",      pipe.get("retriever_top_k", "—")),
            ("Queries",    pipe.get("retriever_query_variants", 1)),
            ("BM25",       green("on") if pipe.get("retriever_with_keywords") else grey("off")),
            ("HyDE",       green("on") if pipe.get("use_hyde")             else grey("off")),
            ("Rewrite",    green("on") if pipe.get("use_query_rewrite")    else grey("off")),
            ("Decompose",  green("on") if pipe.get("use_query_decomposition") else grey("off")),
            ("Rerank",     green("on") if pipe.get("post_use_rerank")      else grey("off")),
            ("Reorder",    green("on") if pipe.get("post_use_reorder")     else grey("off")),
            ("Cutoff",     green(f"on ({pipe.get('post_cutoff')})") if pipe.get("post_use_cutoff") else grey("off")),
            ("Dedup",      green("on") if pipe.get("use_dedup")            else grey("off")),
            ("Consolidate",green("on") if pipe.get("use_llm_consolidation") else grey("off")),
            ("Confidence", green(f"on ({pipe.get('confidence_cutoff')})") if pipe.get("use_confidence_guard") else grey("off")),
        ]
        for label, value in flags:
            print(f"  {label:14s}  {value}")

        # query summary table
        if queries:
            print()
            print(bold(f"  QUERIES  ({len(queries)})"))
            print(f"  {grey(_bar('─', W - 2))}")
            hdr = f"  {'ID':10s}  {'Tokens':>8s}  {'Cost':>10s}  {'Faith':>6s}  {'Relev':>6s}  {'Corr':>6s}  Question"
            print(dim(hdr))
            print(f"  {grey(_bar('─', W - 2))}")
            for q in queries:
                qid     = q.get("query_id", "—")
                tokens  = q.get("tokens", {}).get("total", 0)
                cost    = q.get("costs", {}).get("total", 0.0)
                evals   = q.get("evaluation", {})
                faith   = _score_color(evals.get("faithfulness", {}).get("score"))
                relev   = _score_color(evals.get("relevancy", {}).get("score"))
                corr    = _score_color(evals.get("correctness", {}).get("score"))
                question = _trunc(q.get("query", {}).get("original"), 30)
                refused  = red(" [R]") if q.get("refused") else ""
                print(f"  {qid:10s}  {tokens:>8d}  ${cost:>9.6f}  {faith:>6s}  {relev:>6s}  {corr:>6s}  {question}{refused}")
            print(f"  {grey(_bar('─', W - 2))}")

            # session totals
            total_tokens = sum(q.get("tokens", {}).get("total", 0) for q in queries)
            total_cost   = sum(q.get("costs", {}).get("total", 0.0) for q in queries)
            print(f"  {'TOTAL':10s}  {total_tokens:>8d}  ${total_cost:>9.6f}")

        # optionally print each query in full
        if show_queries:
            for q in queries:
                self.display_query(q, show_nodes=show_nodes,
                                   show_tokens=show_tokens, show_eval=show_eval)

    # ── Display: all sessions comparison ─────────────────────────────────────

    def display_comparison(self):
        sessions = self.load_all_sessions()
        if not sessions:
            print(red("  No sessions found."))
            return

        print()
        print(yellow(_header("SESSION COMPARISON")))
        print()

        col_w = 22
        # header
        headers = ["Session", "Queries", "Avg Tokens", "Avg Cost", "Faith", "Relev", "Corr"]
        widths  = [30, 8, 11, 11, 7, 7, 7]
        header_row = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(dim(header_row))
        print(f"  {grey(_bar('─', W - 2))}")

        for s in sessions:
            cfg     = s["config"]
            queries = s["queries"]
            name    = f"{cfg.get('config_name','')} / {cfg.get('session_name','')}".strip(" /")

            n = len(queries)
            avg_tok  = int(sum(q.get("tokens", {}).get("total", 0) for q in queries) / max(n, 1))
            avg_cost = sum(q.get("costs", {}).get("total", 0.0) for q in queries) / max(n, 1)

            def avg_eval(metric):
                scores = [q.get("evaluation", {}).get(metric, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(metric)]
                scores = [s for s in scores if s is not None]
                return sum(scores) / len(scores) if scores else None

            faith = avg_eval("faithfulness")
            relev = avg_eval("relevancy")
            corr  = avg_eval("correctness")

            row = [
                _trunc(name, widths[0] - 2),
                str(n),
                str(avg_tok),
                f"${avg_cost:.5f}",
                _score_color(faith) if faith is not None else grey("—"),
                _score_color(relev) if relev is not None else grey("—"),
                _score_color(corr)  if corr  is not None else grey("—"),
            ]
            print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))

        print(f"  {grey(_bar('─', W - 2))}")
        print()

    # ── Display: list available sessions ─────────────────────────────────────

    def display_sessions_list(self):
        sessions = self.list_sessions()
        if not sessions:
            print(yellow("  No sessions found in " + self.log_dir))
            return
        print()
        print(cyan(_header("AVAILABLE SESSIONS")))
        for i, s in enumerate(sessions):
            print(f"  {dim(str(i+1).rjust(3))}  {s}")
        print()

    # ─────────────────────────────────────────────────────────────────────────────
    # Latex Paper Stuff
    # ─────────────────────────────────────────────────────────────────────────────

    def export_latex_phase_table(self, session_ids: list[str], caption: str = "", label: str = "") -> str:
        """
        Generates a LaTeX booktabs table comparing configs within a phase.
        Columns: Config | Avg Correctness | Avg Faithfulness | Avg Relevancy | Avg CtxRel | Avg Tokens | Avg Cost
        """
        rows = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            n = len(queries)
            def avg(metric):
                scores = [q.get("evaluation", {}).get(metric, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(metric)]
                scores = [s for s in scores if s is not None]
                return f"{sum(scores)/len(scores):.2f}" if scores else "—"
            avg_tok  = int(sum(q.get("tokens", {}).get("total", 0) for q in queries) / max(n, 1))
            avg_cost = sum(q.get("costs", {}).get("total", 0.0) for q in queries) / max(n, 1)
            rows.append({
                "config":       config.get("config_name", sid),
                "correctness":  avg("correctness"),
                "faithfulness": avg("faithfulness"),
                "relevancy":    avg("relevancy"),
                "ctx_rel":      avg("context_relevancy"),
                "tokens":       str(avg_tok),
                "cost":         f"\\${avg_cost:.5f}",
            })

        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append("Config & Correct. & Faith. & Relev. & Ctx.Rel. & Tokens & Cost (USD) \\\\")
        lines.append("\\midrule")
        for r in rows:
            lines.append(
                f"{r['config']} & {r['correctness']} & {r['faithfulness']} & "
                f"{r['relevancy']} & {r['ctx_rel']} & {r['tokens']} & {r['cost']} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{tab:{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def export_latex_sample_questions(self, session_id: str, question_indices: list[int]) -> str:
        """
        Generates a LaTeX table for 4 sample questions showing
        question, answer, correctness score and feedback.
        question_indices: 0-based indices into the query list
        """
        config, queries = self.load_session(session_id)
        lines = []
        for idx in question_indices:
            if idx >= len(queries):
                continue
            q = queries[idx]
            question = q.get("query", {}).get("original", "")
            answer   = str(q.get("answer", ""))[:200].replace("\n", " ")
            score    = q.get("evaluation", {}).get("correctness", {}).get("score", "—")
            feedback = str(q.get("evaluation", {}).get("correctness", {}).get("feedback", ""))[:150].replace("\n", " ")
            lines.append(f"\\paragraph{{Q: {question}}}")
            lines.append(f"\\textbf{{Antwort:}} {answer}\\\\")
            lines.append(f"\\textbf{{Correctness:}} {score} — {feedback}\\\\")
            lines.append("")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────────
    # Graphics Stuff
    # ─────────────────────────────────────────────────────────────────────────────

    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    def get_sessions_by_phase_prefix(self, prefixes: list[str]) -> list[str]:
        result = []
        for sid in self.list_sessions():
            try:
                config_path = os.path.join(self.log_dir, sid, "config.json")
                with open(config_path) as f:
                    cfg = json.load(f)
                config_name = cfg.get("config_name", "")
                if any(config_name.startswith(p) for p in prefixes):
                    result.append(sid)
            except Exception:
                continue
        return sorted(result)

    def plot_phase_comparison(self, session_ids: list[str], metric: str = "correctness",
                              title: str = "", save_path: str = None):
        """
        Bar chart comparing average metric score across configs in a phase.
        """
        labels, values = [], []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            scores = [q.get("evaluation", {}).get(metric, {}).get("score")
                      for q in queries if q.get("evaluation", {}).get(metric)]
            scores = [s for s in scores if s is not None]
            avg = sum(scores) / len(scores) if scores else 0
            labels.append(config.get("config_name", sid).split("-", 1)[-1])  # strip number prefix
            values.append(avg)

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
        bars = ax.bar(labels, values, color="#4C72B0", edgecolor="white")
        ax.set_ylim(0, 5.5 if metric == "correctness" else 1.1)
        ax.set_ylabel(f"Avg {metric.capitalize()} Score")
        ax.set_title(title or f"Avg {metric.capitalize()} by Config")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_all_metrics_comparison(self, session_ids: list[str],
                                    title: str = "", save_path: str = None):
        """
        Grouped bar chart showing all 4 metrics side by side per config.
        """
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        labels = []
        data = {m: [] for m in metrics}

        for sid in session_ids:
            config, queries = self.load_session(sid)
            labels.append(config.get("config_name", sid).split("-", 1)[-1])
            for m in metrics:
                scores = [q.get("evaluation", {}).get(m, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(m)]
                scores = [s for s in scores if s is not None]
                data[m].append(sum(scores) / len(scores) if scores else 0)

        x = np.arange(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
        for i, (m, color) in enumerate(zip(metrics, colors)):
            # normalise correctness from 0-5 to 0-1 for visual comparison
            vals = [v / 5.0 if m == "correctness" else v for v in data[m]]
            ax.bar(x + i * width, vals, width, label=m.capitalize(), color=color)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (normalised 0–1)")
        ax.set_title(title or "Alle Metriken im Vergleich")
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_per_question_heatmap(self, session_id: str, metric: str = "correctness",
                                  title: str = "", save_path: str = None):
        """
        Heatmap of metric score per question for a single config.
        Good for showing which questions a config struggles with.
        """
        config, queries = self.load_session(session_id)
        questions = [q.get("query", {}).get("original", f"Q{i + 1}")[:40]
                     for i, q in enumerate(queries)]
        scores = [q.get("evaluation", {}).get(metric, {}).get("score") or 0
                  for q in queries]

        fig, ax = plt.subplots(figsize=(12, max(4, len(questions) * 0.35)))
        im = ax.imshow([scores], aspect="auto",
                       cmap="RdYlGn", vmin=0, vmax=5 if metric == "correctness" else 1)
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.3)
        ax.set_title(title or f"{metric.capitalize()} per Question — {config.get('config_name', '')}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_stacked_heatmap(self, session_ids: list[str], question_aliases: dict = None,
                             title: str = "", save_path: str = None):
        """
        Stacked heatmap showing all 4 metrics per question for a single or averaged config.
        Rows = metrics, Columns = questions.
        """
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correctness", "Faithfulness", "Relevancy", "Ctx. Relevancy"]

        # average across sessions if multiple provided
        n_questions = None
        data = {m: None for m in metrics}

        for sid in session_ids:
            _, queries = self.load_session(sid)
            if n_questions is None:
                n_questions = len(queries)
                data = {m: np.zeros(n_questions) for m in metrics}

            for i, q in enumerate(queries):
                for m in metrics:
                    score = q.get("evaluation", {}).get(m, {}).get("score")
                    if score is not None:
                        data[m][i] += float(score)

        n = len(session_ids)
        for m in metrics:
            data[m] = data[m] / n
            # normalise correctness 0-5 to 0-1
            if m == "correctness":
                data[m] = data[m] / 5.0

        matrix = np.array([data[m] for m in metrics])

        if question_aliases:
            xlabels = [f"Q{i}: {question_aliases.get(i, str(i))}" for i in range(n_questions)]
        else:
            xlabels = [f"Q{i}" for i in range(n_questions)]

        fig, ax = plt.subplots(figsize=(max(14, n_questions * 0.6), 3))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_xticks(range(n_questions))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metric_labels, fontsize=9)

        # annotate cells with values
        for row in range(len(metrics)):
            for col in range(n_questions):
                val = matrix[row, col]
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if 0.3 < val < 0.8 else "white")

        plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02,
                     label="Score (normalised 0–1)")
        ax.set_title(title or "Alle Metriken — Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def _filter_questions(self, scores_per_q: list, question_filter: list[int] | None) -> tuple[list, list]:
        """Returns filtered (indices, scores) based on question_filter or aliases keys."""
        if question_filter is None:
            return list(range(len(scores_per_q))), scores_per_q
        indices = [i for i in question_filter if i < len(scores_per_q)]
        filtered = [scores_per_q[i] for i in indices]
        return indices, filtered

    def plot_phase_overview_heatmap(self, session_ids: list[str],
                                    question_aliases: dict = None,
                                    question_filter: list[int] | None = None,
                                    show_cell_text: bool = True,
                                    title: str = "", save_path: str = None):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        short = ["C", "F", "R", "X"]

        if question_filter is None and question_aliases is not None:
            question_filter = sorted(question_aliases.keys())

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            scores_per_q = []
            for q in queries:
                ev = q.get("evaluation", {})
                row = {}
                for m in metrics:
                    s = ev.get(m, {}).get("score")
                    row[m] = float(s) if s is not None else None
                scores_per_q.append(row)
            configs_data.append((name, scores_per_q))

        if question_filter is None:
            q_indices = list(range(len(configs_data[0][1])))
        else:
            q_indices = [i for i in question_filter if i < len(configs_data[0][1])]

        n_configs = len(configs_data)
        n_questions = len(q_indices)

        color_matrix = np.full((n_configs, n_questions), np.nan)
        annot = [[""] * n_questions for _ in range(n_configs)]

        for ci, (name, scores_per_q) in enumerate(configs_data):
            for new_qi, orig_qi in enumerate(q_indices):
                row = scores_per_q[orig_qi]
                vals = []
                parts = []
                for m, s in zip(metrics, short):
                    v = row.get(m)
                    norm = (v / 5.0) if m == "correctness" else v
                    if v is not None:
                        vals.append(norm)
                        parts.append(f"{s}:{v:.1f}" if m == "correctness" else f"{s}:{v:.2f}")
                if vals:
                    color_matrix[ci, new_qi] = sum(vals) / len(vals)
                annot[ci][new_qi] = "\n".join([" ".join(parts[:2]), " ".join(parts[2:])])

        xlabels = [f"Q{i}: {question_aliases.get(i, '')}" if question_aliases else f"Q{i}"
                   for i in q_indices]
        ylabels = [name.split("-", 1)[-1] if "-" in name else name
                   for name, _ in configs_data]

        fig, ax = plt.subplots(figsize=(max(16, n_questions * 0.9), max(4, n_configs * 0.7)))
        im = ax.imshow(color_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_questions))
        ax.set_xticklabels(xlabels, fontsize=6, ha="center")
        ax.set_yticks(range(n_configs))
        ax.set_yticklabels(ylabels, fontsize=8)

        if show_cell_text:
            for ci in range(n_configs):
                for new_qi in range(n_questions):
                    ax.text(new_qi, ci, annot[ci][new_qi], ha="center", va="center",
                            fontsize=4.5, color="black", linespacing=1.3)

        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Normalisierter Gesamtscore (0–1)")
        ax.set_title(title or "Phase — Config × Question Übersicht")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    def plot_phase_question_metric_heatmap(self, session_ids: list[str],
                                           question_aliases: dict = None,
                                           question_filter: list[int] | None = None,
                                           title: str = "", save_path: str = None):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correctness\n(norm. /5)", "Faithfulness", "Relevancy", "Ctx. Relevancy"]

        if question_filter is None and question_aliases is not None:
            question_filter = sorted(question_aliases.keys())

        n_total = None
        acc = None
        cnt = None

        for sid in session_ids:
            _, queries = self.load_session(sid)
            if n_total is None:
                n_total = len(queries)
                acc = {m: np.zeros(n_total) for m in metrics}
                cnt = {m: np.zeros(n_total) for m in metrics}
            for i, q in enumerate(queries):
                ev = q.get("evaluation", {})
                for m in metrics:
                    s = ev.get(m, {}).get("score")
                    if s is not None:
                        acc[m][i] += float(s)
                        cnt[m][i] += 1

        q_indices = question_filter if question_filter is not None else list(range(n_total))
        q_indices = [i for i in q_indices if i < n_total]
        n_questions = len(q_indices)

        matrix = np.zeros((len(metrics), n_questions))
        for mi, m in enumerate(metrics):
            for new_qi, orig_qi in enumerate(q_indices):
                if cnt[m][orig_qi] > 0:
                    v = acc[m][orig_qi] / cnt[m][orig_qi]
                    matrix[mi, new_qi] = (v / 5.0) if m == "correctness" else v

        xlabels = [f"Q{i}: {question_aliases.get(i, '')}" if question_aliases else f"Q{i}"
                   for i in q_indices]

        fig, ax = plt.subplots(figsize=(max(14, n_questions * 0.7), 3.5))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_questions))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metric_labels, fontsize=9)
        for row in range(len(metrics)):
            for col in range(n_questions):
                val = matrix[row, col]
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if 0.25 < val < 0.75 else "white")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Score (norm. 0–1)")
        ax.set_title(title or "Phase — Metrik × Frage (Ø über alle Configs)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def oplot_phase_overview_heatmap_vertical(self, session_ids: list[str],
                                             question_aliases: dict = None,
                                             question_filter: list[int] | None = None,
                                             config_filter: list[int] | None = None,
                                             show_cell_text: bool = True,
                                             config_label: str = "name",
                                             title: str = "", save_path: str = None,
                                             axis_font_scale=0.25,
                                             cell_font_scale=0.18):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        short = ["C", "F", "R", "X"]

        if question_filter is None and question_aliases is not None:
            question_filter = sorted(question_aliases.keys())

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            scores_per_q = []
            for q in queries:
                ev = q.get("evaluation", {})
                row = {}
                for m in metrics:
                    s = ev.get(m, {}).get("score")
                    row[m] = float(s) if s is not None else None
                scores_per_q.append(row)
            configs_data.append((name, scores_per_q))

        # apply config filter by index
        if config_filter is not None:
            configs_data = [configs_data[i] for i in config_filter if i < len(configs_data)]

        if question_filter is None:
            q_indices = list(range(len(configs_data[0][1])))
        else:
            q_indices = [i for i in question_filter if i < len(configs_data[0][1])]

        n_configs = len(configs_data)
        n_questions = len(q_indices)

        color_matrix = np.full((n_questions, n_configs), np.nan)
        annot = [[""] * n_configs for _ in range(n_questions)]

        for ci, (name, scores_per_q) in enumerate(configs_data):
            for new_qi, orig_qi in enumerate(q_indices):
                row = scores_per_q[orig_qi]
                vals = []
                parts = []
                for m, s in zip(metrics, short):
                    v = row.get(m)
                    norm = (v / 5.0) if m == "correctness" else v
                    if v is not None:
                        vals.append(norm)
                        parts.append(f"{s}:{v:.1f}" if m == "correctness" else f"{s}:{v:.2f}")
                if vals:
                    color_matrix[new_qi, ci] = sum(vals) / len(vals)
                annot[new_qi][ci] = "\n".join([" ".join(parts[:2]), " ".join(parts[2:])])

        ylabels = [f"Q{i}: {question_aliases.get(i, '')}" if question_aliases else f"Q{i}"
                   for i in q_indices]

        def _make_xlabel(ci, name):
            if config_label == "number":
                return str(ci + 1)
            elif config_label == "prefix":
                p = name.split("-", 1)
                prefix = p[0] if p[0].isdigit() else str(ci + 1)
                return f"C{prefix}"
            else:
                return name.split("-", 1)[-1] if "-" in name else name

        xlabels = [_make_xlabel(ci, name) for ci, (name, _) in enumerate(configs_data)]

        fig_w = max(6, n_configs * 1.6)
        fig_h = max(10, n_questions * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cell_h_pts = (fig_h / n_questions) * 72
        cell_w_pts = (fig_w / n_configs) * 72
        cell_min_pts = min(cell_h_pts, cell_w_pts)
        axis_fontsize = min(14, max(7, cell_h_pts * axis_font_scale))
        cell_fontsize = min(11, max(5, cell_min_pts * cell_font_scale))

        im = ax.imshow(color_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(n_questions))
        ax.set_yticklabels(ylabels, fontsize=axis_fontsize)
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=axis_fontsize)

        if show_cell_text:
            for new_qi in range(n_questions):
                for ci in range(n_configs):
                    ax.text(ci, new_qi, annot[new_qi][ci], ha="center", va="center",
                            fontsize=cell_fontsize, color="black", linespacing=1.3)

        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Normalisierter Gesamtscore (0–1)")
        ax.set_title(title or "Phase — Question × Config Übersicht")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    def oplot_phase_question_metric_heatmap_vertical(self, session_ids: list[str],
                                                    question_aliases: dict = None,
                                                    question_filter: list[int] | None = None,
                                                    title: str = "", save_path: str = None):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correct.\n(/5)", "Faith.", "Relev.", "Ctx.Rel."]

        if question_filter is None and question_aliases is not None:
            question_filter = sorted(question_aliases.keys())

        n_total = None
        acc = None
        cnt = None

        for sid in session_ids:
            _, queries = self.load_session(sid)
            if n_total is None:
                n_total = len(queries)
                acc = {m: np.zeros(n_total) for m in metrics}
                cnt = {m: np.zeros(n_total) for m in metrics}
            for i, q in enumerate(queries):
                ev = q.get("evaluation", {})
                for m in metrics:
                    s = ev.get(m, {}).get("score")
                    if s is not None:
                        acc[m][i] += float(s)
                        cnt[m][i] += 1

        q_indices = question_filter if question_filter is not None else list(range(n_total))
        q_indices = [i for i in q_indices if i < n_total]
        n_questions = len(q_indices)

        matrix = np.zeros((n_questions, len(metrics)))
        for mi, m in enumerate(metrics):
            for new_qi, orig_qi in enumerate(q_indices):
                if cnt[m][orig_qi] > 0:
                    v = acc[m][orig_qi] / cnt[m][orig_qi]
                    matrix[new_qi, mi] = (v / 5.0) if m == "correctness" else v

        ylabels = [f"Q{i}: {question_aliases.get(i, '')}" if question_aliases else f"Q{i}"
                   for i in q_indices]

        fig, ax = plt.subplots(figsize=(5, max(10, n_questions * 0.55)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(n_questions))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, fontsize=9)
        for new_qi in range(n_questions):
            for mi in range(len(metrics)):
                val = matrix[new_qi, mi]
                ax.text(mi, new_qi, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.25 < val < 0.75 else "white")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Score (norm. 0–1)")
        ax.set_title(title or "Phase — Frage × Metrik (Ø über alle Configs)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_phase_bar_chart(self, session_ids: list[str],
                             config_label: str = "name",
                             title: str = "", save_path: str = None,
                             axis_font_scale: float = 0.20,
                             bar_label_scale: float = 0.15):
        """
        Grouped bar chart: one group per config, 4 bars per group (one per metric).
        Correctness normalised to 0-1 for visual comparison.
        """
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correctness (/5)", "Faithfulness", "Relevancy", "Ctx. Relevancy"]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            avgs = {}
            for m in metrics:
                scores = [q.get("evaluation", {}).get(m, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(m)]
                scores = [float(s) for s in scores if s is not None]
                avgs[m] = (sum(scores) / len(scores)) if scores else 0.0
            configs_data.append((name, avgs))

        def _make_xlabel(ci, name):
            if config_label == "number":
                return str(ci + 1)
            elif config_label == "prefix":
                p = name.split("-", 1)
                prefix = p[0] if p[0].isdigit() else str(ci + 1)
                return f"C{prefix}"
            else:
                return name.split("-", 1)[-1] if "-" in name else name

        xlabels = [_make_xlabel(ci, name) for ci, (name, _) in enumerate(configs_data)]
        x = np.arange(len(configs_data))
        width = 0.18

        fig_w = max(10, len(configs_data) * 1.4)
        fig_h = 5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # dynamic font scaling based on figure size
        cell_w_pts = (fig_w / len(configs_data)) * 72
        cell_h_pts = fig_h * 72
        cell_min_pts = min(cell_w_pts, cell_h_pts)
        axis_fontsize = min(14, max(7, cell_w_pts * axis_font_scale))
        bar_label_fontsize = min(10, max(5, cell_min_pts * bar_label_scale))

        for mi, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [(avgs[m] / 5.0) if m == "correctness" else avgs[m]
                    for _, avgs in configs_data]
            bars = ax.bar(x + mi * width, vals, width, label=label, color=color)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=bar_label_fontsize, rotation=90)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=axis_fontsize)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (normalisiert 0–1)", fontsize=axis_fontsize)
        ax.set_title(title or "Phase — Ø Evaluationsmetriken pro Config",
                     fontsize=axis_fontsize * 1.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  ncol=len(metric_labels), frameon=False,
                  fontsize=axis_fontsize * 0.9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def oplot_phase_bar_chart_horizontal(self, session_ids: list[str],
                                        config_label: str = "name",
                                        title: str = "", save_path: str = None,
                                        axis_font_scale: float = 0.20,
                                        bar_label_scale: float = 0.15):
        """
        Horizontal grouped bar chart: one group per config, 4 bars per group.
        Correctness normalised to 0-1 for visual comparison.
        """
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correctness (/5)", "Faithfulness", "Relevancy", "Ctx. Relevancy"]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            avgs = {}
            for m in metrics:
                scores = [q.get("evaluation", {}).get(m, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(m)]
                scores = [float(s) for s in scores if s is not None]
                avgs[m] = (sum(scores) / len(scores)) if scores else 0.0
            configs_data.append((name, avgs))

        def _make_ylabel(ci, name):
            if config_label == "number":
                return str(ci + 1)
            elif config_label == "prefix":
                p = name.split("-", 1)
                prefix = p[0] if p[0].isdigit() else str(ci + 1)
                return f"C{prefix}"
            else:
                return name.split("-", 1)[-1] if "-" in name else name

        ylabels = [_make_ylabel(ci, name) for ci, (name, _) in enumerate(configs_data)]
        y = np.arange(len(configs_data))
        height = 0.18

        fig_h = max(6, len(configs_data) * 0.8)
        fig_w = 10
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # dynamic font scaling
        cell_h_pts = (fig_h / len(configs_data)) * 72
        cell_w_pts = fig_w * 72
        cell_min_pts = min(cell_h_pts, cell_w_pts)
        axis_fontsize = min(14, max(7, cell_h_pts * axis_font_scale))
        bar_label_fontsize = min(10, max(5, cell_min_pts * bar_label_scale))

        for mi, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [(avgs[m] / 5.0) if m == "correctness" else avgs[m]
                    for _, avgs in configs_data]
            bars = ax.barh(y + mi * height, vals, height, label=label, color=color)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}", ha="left", va="center",
                        fontsize=bar_label_fontsize)

        ax.set_yticks(y + height * 1.5)
        ax.set_yticklabels(ylabels, fontsize=axis_fontsize)
        ax.set_xlim(0, 1.45)
        ax.set_xlabel("Score (normalisiert 0–1)", fontsize=axis_fontsize)
        ax.set_title(title or "Phase — Ø Evaluationsmetriken pro Config",
                     fontsize=axis_fontsize * 1.2)
        ax.legend(loc="upper right", fontsize=axis_fontsize * 0.9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def export_latex_eval_table(self, session_ids: list[str],
                                caption: str = "", label: str = "",
                                save_path: str = None) -> str:
        """
        LaTeX booktabs table: Config | Avg Correctness | Faithfulness | Relevancy | Ctx.Rel.
        """
        rows = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            n = len(queries)

            def avg(metric):
                scores = [q.get("evaluation", {}).get(metric, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(metric)]
                scores = [float(s) for s in scores if s is not None]
                return f"{sum(scores) / len(scores):.2f}" if scores else "—"

            rows.append({
                "config": config.get("config_name", sid),
                "correctness": avg("correctness"),
                "faithfulness": avg("faithfulness"),
                "relevancy": avg("relevancy"),
                "ctx_rel": avg("context_relevancy"),
            })

        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Config & Correct. & Faith. & Relev. & Ctx.Rel. \\\\")
        lines.append("\\midrule")
        for r in rows:
            lines.append(
                f"{r['config']} & {r['correctness']} & {r['faithfulness']} & "
                f"{r['relevancy']} & {r['ctx_rel']} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{tab:{label}}}")
        lines.append("\\end{table}")

        result = "\n".join(lines)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result)
        return result

    def export_latex_cost_table(self, session_ids: list[str],
                                caption: str = "", label: str = "",
                                save_path: str = None) -> str:
        """
        LaTeX booktabs table: Config | Avg Tokens | Avg Calls | Avg Cost (USD) | Avg Time (s)
        """
        rows = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            n = max(len(queries), 1)

            avg_tokens = int(sum(q.get("tokens", {}).get("total", 0) for q in queries) / n)
            avg_calls = sum(q.get("tokens", {}).get("total_calls", 0) for q in queries) / n
            avg_cost = sum(q.get("costs", {}).get("total", 0.0) for q in queries) / n
            avg_time = sum(q.get("timings", {}).get("total_s", 0.0)
                           for q in queries if q.get("timings", {}).get("total_s") is not None) / n

            rows.append({
                "config": config.get("config_name", sid),
                "tokens": str(avg_tokens),
                "calls": f"{avg_calls:.1f}",
                "cost": f"\\${avg_cost:.5f}",
                "time": f"{avg_time:.1f}s",
            })

        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Config & Ø Tokens & Ø Calls & Ø Kosten (USD) & Ø Zeit \\\\")
        lines.append("\\midrule")
        for r in rows:
            lines.append(
                f"{r['config']} & {r['tokens']} & {r['calls']} & "
                f"{r['cost']} & {r['time']} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{tab:{label}}}")
        lines.append("\\end{table}")

        result = "\n".join(lines)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result)
        return result

    def oplot_index_build_stats(self, title: str = "", save_path: str = None,
                               axis_font_scale: float = 0.25,
                               bar_label_scale: float = 0.18):
        """
        Bar chart showing index build time, size and chunk count
        for BASE-...-index-build sessions.
        Reads index_time, index_size_mb and chunks_in_store from config.json.
        """
        build_sessions = [s for s in self.list_sessions() if "index-build" in s.lower()
                          or s.startswith("BASE-")]

        data = []
        for sid in build_sessions:
            try:
                config_path = os.path.join(self.log_dir, sid, "config.json")
                with open(config_path) as f:
                    cfg = json.load(f)
                index_info = cfg.get("index", {})
                name = cfg.get("config_name", sid)
                time_str = index_info.get("index_time", "0s")
                size_mb = index_info.get("index_size_mb", 0.0)
                chunks = index_info.get("chunks_in_store", 0)
                try:
                    time_s = float(str(time_str).rstrip("s"))
                except ValueError:
                    time_s = 0.0
                data.append({
                    "name": name,
                    "time_s": time_s,
                    "size_mb": float(size_mb),
                    "chunks": int(chunks),
                })
            except Exception:
                continue

        if not data:
            print("No index build sessions found.")
            return

        data.sort(key=lambda d: d["time_s"])
        names = [d["name"].split("-", 1)[-1] if "-" in d["name"] else d["name"] for d in data]
        times = [d["time_s"] for d in data]
        sizes = [d["size_mb"] for d in data]
        chunks = [d["chunks"] for d in data]

        fig_h = max(5, len(data) * 0.6)
        fig_w = 14
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

        cell_h_pts = (fig_h / len(data)) * 72
        cell_w_pts = (fig_w / 3) * 72
        cell_min_pts = min(cell_h_pts, cell_w_pts)
        axis_fontsize = min(13, max(7, cell_h_pts * axis_font_scale))
        bar_label_fontsize = min(11, max(5, cell_min_pts * bar_label_scale))

        y = np.arange(len(data))

        # Time
        axes[0].barh(y, times, color="#4C72B0")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names, fontsize=axis_fontsize)
        axes[0].set_xlim(0, max(times) * 1.15)
        axes[0].set_xlabel("Indexierungszeit (s)", fontsize=axis_fontsize)
        axes[0].set_title("Build-Zeit", fontsize=axis_fontsize * 1.1)
        for i, v in enumerate(times):
            axes[0].text(v + max(times) * 0.01, i, f"{v:.0f}s",
                         va="center", fontsize=bar_label_fontsize)

        # Size
        axes[1].barh(y, sizes, color="#55A868")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([], fontsize=axis_fontsize)
        axes[1].set_xlim(0, max(sizes) * 1.25)
        axes[1].set_xlabel("Indexgröße (MB)", fontsize=axis_fontsize)
        axes[1].set_title("Indexgröße", fontsize=axis_fontsize * 1.1)
        for i, v in enumerate(sizes):
            axes[1].text(v + max(sizes) * 0.01, i, f"{v:.1f} MB",
                         va="center", fontsize=bar_label_fontsize)

        # Chunks
        axes[2].barh(y, chunks, color="#DD8452")
        axes[2].set_yticks(y)
        axes[2].set_yticklabels([], fontsize=axis_fontsize)
        axes[2].set_xlim(0, max(chunks) * 1.2)
        axes[2].set_xlabel("Anzahl Chunks", fontsize=axis_fontsize)
        axes[2].set_title("Chunks im Index", fontsize=axis_fontsize * 1.1)
        for i, v in enumerate(chunks):
            axes[2].text(v + max(chunks) * 0.01, i, f"{v:,}",
                         va="center", fontsize=bar_label_fontsize)

        fig.suptitle(title or "Index Build-Statistiken",
                     fontsize=axis_fontsize * 1.3, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def _save_tex(self, content: str, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def export_latex_figure_overview(self, img_path: str, caption: str, label: str,
                                     save_path: str = None) -> str:
        tex = (
            f"\\begin{{figure}}[H]\n"
            f"    \\centering\n"
            f"    \\makebox[\\textwidth][c]{{\n"
            f"        \\includegraphics[height=0.95\\textheight]{{{img_path}}}\n"
            f"    }}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{fig:{label}}}\n"
            f"\\end{{figure}}\n"
        )
        if save_path:
            self._save_tex(tex, save_path)
        return tex

    def export_latex_figure_metrics(self, img_path_a: str, img_path_b: str,
                                    caption: str, label: str,
                                    caption_a: str = "Q0--Q14",
                                    caption_b: str = "Q15--Q29",
                                    save_path: str = None) -> str:
        tex = (
            f"\\begin{{figure}}[H]\n"
            f"    \\centering\n"
            f"    \\begin{{subfigure}}[t]{{0.48\\textwidth}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{img_path_a}}}\n"
            f"        \\caption{{{caption_a}}}\n"
            f"        \\label{{fig:{label}_a}}\n"
            f"    \\end{{subfigure}}\n"
            f"    \\hfill\n"
            f"    \\begin{{subfigure}}[t]{{0.48\\textwidth}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{img_path_b}}}\n"
            f"        \\caption{{{caption_b}}}\n"
            f"        \\label{{fig:{label}_b}}\n"
            f"    \\end{{subfigure}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{fig:{label}}}\n"
            f"\\end{{figure}}\n"
        )
        if save_path:
            self._save_tex(tex, save_path)
        return tex

    def export_latex_figure_bar(self, img_path: str, caption: str, label: str,
                                save_path: str = None) -> str:
        tex = (
            f"\\begin{{figure}}[H]\n"
            f"    \\centering\n"
            f"    \\includegraphics[width=\\textwidth]{{{img_path}}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{fig:{label}}}\n"
            f"\\end{{figure}}\n"
        )
        if save_path:
            self._save_tex(tex, save_path)
        return tex

    def export_latex_figure_index_build(self, img_path: str, caption: str, label: str,
                                        width: str = "1.2\\textwidth",
                                        save_path: str = None) -> str:
        tex = (
            f"\\begin{{figure}}[H]\n"
            f"    \\centering\n"
            f"    \\makebox[\\textwidth][c]{{\n"
            f"        \\includegraphics[width={width}]{{{img_path}}}\n"
            f"    }}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{fig:{label}}}\n"
            f"\\end{{figure}}\n"
        )
        if save_path:
            self._save_tex(tex, save_path)
        return tex



    def plot_phase_overview_heatmap_vertical(self, session_ids: list[str],
                                             question_aliases: dict = None,
                                             question_filter: list[int] | None = None,
                                             config_filter: list[int] | None = None,
                                             show_cell_text: bool = True,
                                             config_label: str = "name",
                                             title: str = "", save_path: str = None,
                                             axis_font_scale=0.25,
                                             cell_font_scale=0.18):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        short = ["C", "F", "R", "X"]

        if question_filter is None and question_aliases is not None:
            question_filter = sorted(question_aliases.keys())

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            scores_per_q = []
            for q in queries:
                ev = q.get("evaluation", {})
                row = {}
                for m in metrics:
                    s = ev.get(m, {}).get("score")
                    row[m] = float(s) if s is not None else None
                scores_per_q.append(row)
            configs_data.append((name, scores_per_q))

        if config_filter is not None:
            configs_data = [configs_data[i] for i in config_filter if i < len(configs_data)]

        if question_filter is None:
            q_indices = list(range(len(configs_data[0][1])))
        else:
            q_indices = [i for i in question_filter if i < len(configs_data[0][1])]

        n_configs = len(configs_data)
        n_questions = len(q_indices)

        color_matrix = np.full((n_questions, n_configs), np.nan)
        annot = [[""] * n_configs for _ in range(n_questions)]

        for ci, (name, scores_per_q) in enumerate(configs_data):
            for new_qi, orig_qi in enumerate(q_indices):
                row = scores_per_q[orig_qi]
                vals = []
                parts = []
                for m, s in zip(metrics, short):
                    v = row.get(m)
                    norm = (v / 5.0) if m == "correctness" else v
                    if v is not None:
                        vals.append(norm)
                        parts.append(f"{s}:{v:.1f}" if m == "correctness" else f"{s}:{v:.2f}")
                if vals:
                    color_matrix[new_qi, ci] = sum(vals) / len(vals)
                annot[new_qi][ci] = "\n".join([" ".join(parts[:2]), " ".join(parts[2:])])

        ylabels = [f"Q{i}: {question_aliases.get(i, '')}" if question_aliases else f"Q{i}"
                   for i in q_indices]

        def _make_xlabel(ci, name):
            if config_label == "number":
                return str(ci + 1)
            elif config_label == "prefix":
                p = name.split("-", 1)
                prefix = p[0] if p[0].isdigit() else str(ci + 1)
                return f"C{prefix}"
            else:
                return name.split("-", 1)[-1] if "-" in name else name

        xlabels = [_make_xlabel(ci, name) for ci, (name, _) in enumerate(configs_data)]

        fig_w = max(6, n_configs * 1.6)
        fig_h = max(10, n_questions * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cell_h_pts = (fig_h / n_questions) * 72
        cell_w_pts = (fig_w / n_configs) * 72
        cell_min_pts = min(cell_h_pts, cell_w_pts)
        axis_fontsize = min(14, max(7, cell_h_pts * axis_font_scale))
        cell_fontsize = min(11, max(5, cell_min_pts * cell_font_scale))

        im = ax.imshow(color_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(n_questions))
        ax.set_yticklabels(ylabels, fontsize=axis_fontsize)
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=axis_fontsize)

        if show_cell_text:
            for new_qi in range(n_questions):
                for ci in range(n_configs):
                    ax.text(ci, new_qi, annot[new_qi][ci], ha="center", va="center",
                            fontsize=cell_fontsize, color="black", linespacing=1.3)

        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Normalisierter Gesamtscore (0–1)")
        wrapped_title = "\n".join(textwrap.wrap(title or "Phase — Question × Config Übersicht", width=80))
        ax.set_title(wrapped_title, fontsize=axis_fontsize * 1.1)
        fig.subplots_adjust(top=0.92)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    def plot_phase_question_metric_heatmap_vertical(self, session_ids: list[str],
                                                    question_aliases: dict = None,
                                                    question_filter: list[int] | None = None,
                                                    title: str = "", save_path: str = None):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correct.\n(/5)", "Faith.", "Relev.", "Ctx.Rel."]

        if question_filter is None and question_aliases is not None:
            question_filter = sorted(question_aliases.keys())

        n_total = None
        acc = None
        cnt = None

        for sid in session_ids:
            _, queries = self.load_session(sid)
            if n_total is None:
                n_total = len(queries)
                acc = {m: np.zeros(n_total) for m in metrics}
                cnt = {m: np.zeros(n_total) for m in metrics}
            for i, q in enumerate(queries):
                ev = q.get("evaluation", {})
                for m in metrics:
                    s = ev.get(m, {}).get("score")
                    if s is not None:
                        acc[m][i] += float(s)
                        cnt[m][i] += 1

        q_indices = question_filter if question_filter is not None else list(range(n_total))
        q_indices = [i for i in q_indices if i < n_total]
        n_questions = len(q_indices)

        matrix = np.zeros((n_questions, len(metrics)))
        for mi, m in enumerate(metrics):
            for new_qi, orig_qi in enumerate(q_indices):
                if cnt[m][orig_qi] > 0:
                    v = acc[m][orig_qi] / cnt[m][orig_qi]
                    matrix[new_qi, mi] = (v / 5.0) if m == "correctness" else v

        ylabels = [f"Q{i}: {question_aliases.get(i, '')}" if question_aliases else f"Q{i}"
                   for i in q_indices]

        fig, ax = plt.subplots(figsize=(5, max(10, n_questions * 0.55)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(n_questions))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, fontsize=9)
        for new_qi in range(n_questions):
            for mi in range(len(metrics)):
                val = matrix[new_qi, mi]
                ax.text(mi, new_qi, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.25 < val < 0.75 else "white")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Score (norm. 0–1)")
        wrapped_title = "\n".join(textwrap.wrap(title or "Phase — Frage × Metrik (Ø über alle Configs)", width=50))
        ax.set_title(wrapped_title, fontsize=9)
        fig.subplots_adjust(top=0.92)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_phase_bar_chart_horizontal(self, session_ids: list[str],
                                        config_label: str = "name",
                                        title: str = "", save_path: str = None,
                                        axis_font_scale: float = 0.20,
                                        bar_label_scale: float = 0.15):
        metrics = ["correctness", "faithfulness", "relevancy", "context_relevancy"]
        metric_labels = ["Correctness (/5)", "Faithfulness", "Relevancy", "Ctx. Relevancy"]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            avgs = {}
            for m in metrics:
                scores = [q.get("evaluation", {}).get(m, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(m)]
                scores = [float(s) for s in scores if s is not None]
                avgs[m] = (sum(scores) / len(scores)) if scores else 0.0
            configs_data.append((name, avgs))

        def _make_ylabel(ci, name):
            if config_label == "number":
                return str(ci + 1)
            elif config_label == "prefix":
                p = name.split("-", 1)
                prefix = p[0] if p[0].isdigit() else str(ci + 1)
                return f"C{prefix}"
            else:
                return name.split("-", 1)[-1] if "-" in name else name

        ylabels = [_make_ylabel(ci, name) for ci, (name, _) in enumerate(configs_data)]
        y = np.arange(len(configs_data))
        height = 0.18

        fig_h = max(6, len(configs_data) * 0.8)
        fig_w = 10
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cell_h_pts = (fig_h / len(configs_data)) * 72
        cell_w_pts = fig_w * 72
        cell_min_pts = min(cell_h_pts, cell_w_pts)
        axis_fontsize = min(14, max(7, cell_h_pts * axis_font_scale))
        bar_label_fontsize = min(10, max(5, cell_min_pts * bar_label_scale))

        for mi, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [(avgs[m] / 5.0) if m == "correctness" else avgs[m]
                    for _, avgs in configs_data]
            bars = ax.barh(y + mi * height, vals, height, label=label, color=color)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}", ha="left", va="center",
                        fontsize=bar_label_fontsize)

        ax.set_yticks(y + height * 1.5)
        ax.set_yticklabels(ylabels, fontsize=axis_fontsize)
        ax.set_xlim(0, 1.45)
        ax.set_xlabel("Score (normalisiert 0–1)", fontsize=axis_fontsize)
        wrapped_title = "\n".join(textwrap.wrap(title or "Phase — Ø Evaluationsmetriken pro Config", width=70))
        ax.set_title(wrapped_title, fontsize=axis_fontsize * 1.2)
        fig.subplots_adjust(top=0.88)
        ax.legend(loc="upper right", fontsize=axis_fontsize * 0.9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_index_build_stats(self, title: str = "", save_path: str = None,
                               axis_font_scale: float = 0.25,
                               bar_label_scale: float = 0.18):
        build_sessions = [s for s in self.list_sessions() if "index-build" in s.lower()
                          or s.startswith("BASE-")]

        data = []
        for sid in build_sessions:
            try:
                config_path = os.path.join(self.log_dir, sid, "config.json")
                with open(config_path) as f:
                    cfg = json.load(f)
                index_info = cfg.get("index", {})
                name = cfg.get("config_name", sid)
                time_str = index_info.get("index_time", "0s")
                size_mb = index_info.get("index_size_mb", 0.0)
                chunks = index_info.get("chunks_in_store", 0)
                try:
                    time_s = float(str(time_str).rstrip("s"))
                except ValueError:
                    time_s = 0.0
                data.append({
                    "name": name,
                    "time_s": time_s,
                    "size_mb": float(size_mb),
                    "chunks": int(chunks),
                })
            except Exception:
                continue

        if not data:
            print("No index build sessions found.")
            return

        data.sort(key=lambda d: d["time_s"])
        names = [d["name"].split("-", 1)[-1] if "-" in d["name"] else d["name"] for d in data]
        times = [d["time_s"] for d in data]
        sizes = [d["size_mb"] for d in data]
        chunks = [d["chunks"] for d in data]

        fig_h = max(5, len(data) * 0.6)
        fig_w = 14
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

        cell_h_pts = (fig_h / len(data)) * 72
        cell_w_pts = (fig_w / 3) * 72
        cell_min_pts = min(cell_h_pts, cell_w_pts)
        axis_fontsize = min(13, max(7, cell_h_pts * axis_font_scale))
        bar_label_fontsize = min(11, max(5, cell_min_pts * bar_label_scale))

        y = np.arange(len(data))

        axes[0].barh(y, times, color="#4C72B0")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names, fontsize=axis_fontsize)
        axes[0].set_xlim(0, max(times) * 1.15)
        axes[0].set_xlabel("Indexierungszeit (s)", fontsize=axis_fontsize)
        axes[0].set_title("Build-Zeit", fontsize=axis_fontsize * 1.1)
        for i, v in enumerate(times):
            axes[0].text(v + max(times) * 0.01, i, f"{v:.0f}s",
                         va="center", fontsize=bar_label_fontsize)

        axes[1].barh(y, sizes, color="#55A868")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([], fontsize=axis_fontsize)
        axes[1].set_xlim(0, max(sizes) * 1.25)
        axes[1].set_xlabel("Indexgröße (MB)", fontsize=axis_fontsize)
        axes[1].set_title("Indexgröße", fontsize=axis_fontsize * 1.1)
        for i, v in enumerate(sizes):
            axes[1].text(v + max(sizes) * 0.01, i, f"{v:.1f} MB",
                         va="center", fontsize=bar_label_fontsize)

        axes[2].barh(y, chunks, color="#DD8452")
        axes[2].set_yticks(y)
        axes[2].set_yticklabels([], fontsize=axis_fontsize)
        axes[2].set_xlim(0, max(chunks) * 1.2)
        axes[2].set_xlabel("Anzahl Chunks", fontsize=axis_fontsize)
        axes[2].set_title("Chunks im Index", fontsize=axis_fontsize * 1.1)
        for i, v in enumerate(chunks):
            axes[2].text(v + max(chunks) * 0.01, i, f"{v:,}",
                         va="center", fontsize=bar_label_fontsize)

        wrapped_title = "\n".join(textwrap.wrap(title or "Index Build-Statistiken", width=80))
        fig.suptitle(wrapped_title, fontsize=axis_fontsize * 1.3, fontweight="bold")
        fig.subplots_adjust(top=0.88)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_metric_boxplot(self, session_ids: list[str],
                            metric: str = "correctness",
                            config_label: str = "prefix",
                            title: str = "", save_path: str = None,
                            axis_font_scale: float = 0.20):
        """
        Horizontal boxplot per config for a single metric.
        Configs sorted by median score.
        """
        import textwrap

        metric_display = {
            "correctness": "Correctness (1–5)",
            "faithfulness": "Faithfulness (0–1)",
            "relevancy": "Relevancy (0–1)",
            "context_relevancy": "Ctx. Relevancy (0–1)",
        }

        configs_data = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            name = config.get("config_name", sid)
            scores = []
            for q in queries:
                s = q.get("evaluation", {}).get(metric, {}).get("score")
                if s is not None:
                    scores.append(float(s))
            if scores:
                configs_data.append((name, scores))

        if not configs_data:
            print(f"No data for metric: {metric}")
            return

        # sort by median
        configs_data.sort(key=lambda x: np.median(x[1]))

        def _make_label(name):
            if config_label == "prefix":
                p = name.split("-", 1)
                prefix = p[0] if p[0].isdigit() else name
                return f"C{prefix}"
            elif config_label == "number":
                return str(configs_data.index((name, configs_data[[c[0] for c in configs_data].index(name)][1])) + 1)
            else:
                return name.split("-", 1)[-1] if "-" in name else name

        labels = [_make_label(name) for name, _ in configs_data]
        data = [scores for _, scores in configs_data]

        fig_h = max(6, len(configs_data) * 0.45)
        fig_w = 10
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cell_h_pts = (fig_h / len(configs_data)) * 72
        axis_fontsize = min(13, max(7, cell_h_pts * axis_font_scale))

        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor="#C44E52", alpha=0.7))

        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=axis_fontsize)

        # set x range based on metric
        if metric == "correctness":
            ax.set_xlim(0.5, 5.5)
            ax.set_xlabel("Score (1–5)", fontsize=axis_fontsize)
        else:
            ax.set_xlim(-0.05, 1.1)
            ax.set_xlabel("Score (0–1)", fontsize=axis_fontsize)

        ax.axvline(x=np.median([s for _, scores in configs_data for s in scores]),
                   color="grey", linestyle="--", linewidth=1, alpha=0.5,
                   label="Gesamtmedian")
        ax.legend(fontsize=axis_fontsize * 0.85)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        wrapped = "\n".join(textwrap.wrap(
            title or f"Alle Konfigurationen — {metric_display.get(metric, metric)}",
            width=70))
        ax.set_title(wrapped, fontsize=axis_fontsize * 1.15)
        fig.subplots_adjust(top=0.92)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def export_latex_ranking_table(self, session_ids: list[str],
                                   sort_by: str = "correctness",
                                   caption: str = "", label: str = "",
                                   save_path: str = None) -> str:
        """
        Summary ranking table: all configs sorted by avg of sort_by metric.
        Columns: Rank | Config | Correct. | Faith. | Relev. | Ctx.Rel. | Ø Cost | Ø Zeit
        """
        rows = []
        for sid in session_ids:
            config, queries = self.load_session(sid)
            n = max(len(queries), 1)
            name = config.get("config_name", sid)

            def avg(metric):
                scores = [q.get("evaluation", {}).get(metric, {}).get("score")
                          for q in queries if q.get("evaluation", {}).get(metric)]
                scores = [float(s) for s in scores if s is not None]
                return sum(scores) / len(scores) if scores else 0.0

            avg_cost = sum(q.get("costs", {}).get("total", 0.0) for q in queries) / n
            avg_time = sum(q.get("timings", {}).get("total_s", 0.0)
                           for q in queries
                           if q.get("timings", {}).get("total_s") is not None) / n

            rows.append({
                "config": name,
                "correctness": avg("correctness"),
                "faithfulness": avg("faithfulness"),
                "relevancy": avg("relevancy"),
                "ctx_rel": avg("context_relevancy"),
                "cost": avg_cost,
                "time": avg_time,
            })

        rows.sort(key=lambda r: r[sort_by], reverse=True)

        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{rllcccccc}")
        lines.append("\\toprule")
        lines.append("Rang & Config & Correct. & Faith. & Relev. & Ctx.Rel. & Ø Kosten & Ø Zeit \\\\")
        lines.append("\\midrule")

        for rank, r in enumerate(rows, 1):
            is_best = rank == 1

            def fmt(val):
                return f"\\textbf{{{val}}}" if is_best else str(val)

            correctness = f"{r['correctness']:.2f}"
            faithfulness = f"{r['faithfulness']:.2f}"
            relevancy = f"{r['relevancy']:.2f}"
            ctx_rel = f"{r['ctx_rel']:.2f}"
            cost = f"\\${r['cost']:.5f}"
            time = f"{r['time']:.1f}s"

            lines.append(
                f"{fmt(rank)} & "
                f"{fmt(r['config'])} & "
                f"{fmt(correctness)} & "
                f"{fmt(faithfulness)} & "
                f"{fmt(relevancy)} & "
                f"{fmt(ctx_rel)} & "
                f"{fmt(cost)} & "
                f"{fmt(time)} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{tab:{label}}}")
        lines.append("\\end{table}")

        result = "\n".join(lines)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result)
        return result

    def export_latex_config_table(self, session_id: str,
                                  phase_description: str = "",
                                  save_path: str = None) -> str:
        """
        Full parameter table for a single config session.
        Groups: Modelle, Indexierung, Retrieval, Postprocessing,
                Query-Transformation, Kontextverarbeitung, Generierung, Guards
        Non-default values are highlighted in bold.
        """
        config, _ = self.load_session(session_id)
        pipe = config.get("pipeline", {})
        prompt = config.get("prompt", {})
        name = config.get("config_name", session_id)

        # defaults matching RAGPipeline.__init__
        DEFAULTS = {
            "rag_llm_model": "mistralai/mistral-7b-instruct-v0.1",
            "meta_llm_model": "openai/gpt-4o-mini",
            "meta_temperature": 0.0,
            "llm_max_response": 512,
            "llm_context_window": 4096,
            "llm_temperature": 0.1,
            "llm_top_p": 0.9,
            "llm_top_k": 40,
            "llm_instruction": "",
            "embed_model": "BAAI/bge-small-en-v1.5",
            "chunk_size": 512,
            "chunk_overlay": 50,
            "embed_with_markdown": False,
            "embed_split_on": "tokens",
            "special_mode": "none",
            "retriever_top_k": 10,
            "retriever_query_variants": 1,
            "retriever_with_keywords": False,
            "post_use_cutoff": False,
            "post_cutoff": 0.14,
            "post_use_rerank": False,
            "post_rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "post_rerank_top_n": 3,
            "post_use_reorder": False,
            "response_mode": "COMPACT",
            "use_hyde": False,
            "use_query_rewrite": False,
            "use_query_decomposition": False,
            "use_dedup": False,
            "dedup_threshold": 0.85,
            "use_llm_consolidation": False,
            "use_confidence_guard": False,
            "confidence_cutoff": 0.5,
        }

        def val(key, source=pipe):
            return source.get(key, DEFAULTS.get(key, "—"))

        def fmt(key, source=pipe):
            v = val(key, source)
            d = DEFAULTS.get(key)
            v_str = str(v)
            # escape special latex chars
            v_str = v_str.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
            if d is not None and v != d:
                return f"\\textbf{{{v_str}}}"
            return v_str

        def bool_fmt(key, source=pipe):
            v = val(key, source)
            d = DEFAULTS.get(key)
            v_str = "Ja" if v else "Nein"
            if d is not None and v != d:
                return f"\\textbf{{{v_str}}}"
            return v_str

        groups = []

        # ── Modelle ──────────────────────────────────────────────────────────────
        groups.append(("Modelle", [
            ("Generator LLM", fmt("rag_llm_model")),
            ("Meta / Eval LLM", fmt("meta_llm_model")),
            ("Embedding-Modell", fmt("embed_model")),
            ("LLM Temperatur", fmt("llm_temperature")),
            ("Meta Temperatur", fmt("meta_temperature")),
            ("Max Response Tokens", fmt("llm_max_response")),
            ("Context Window", fmt("llm_context_window")),
            ("Top-p", fmt("llm_top_p")),
            ("Top-k", fmt("llm_top_k")),
        ]))

        # ── Indexierung ───────────────────────────────────────────────────────────
        groups.append(("Indexierung", [
            ("Chunking-Modus", fmt("special_mode")),
            ("Split-Strategie", fmt("embed_split_on")),
            ("Markdown-Vorparser", bool_fmt("embed_with_markdown")),
            ("Chunk-Größe", fmt("chunk_size")),
            ("Chunk-Überlappung", fmt("chunk_overlay")),
        ]))

        # ── Retrieval ─────────────────────────────────────────────────────────────
        groups.append(("Retrieval", [
            ("top-k", fmt("retriever_top_k")),
            ("Query-Varianten (Fusion)", fmt("retriever_query_variants")),
            ("BM25 Keyword-Retrieval", bool_fmt("retriever_with_keywords")),
        ]))

        # ── Postprocessing ────────────────────────────────────────────────────────
        groups.append(("Postprocessing", [
            ("Score-Cutoff aktiv", bool_fmt("post_use_cutoff")),
            ("Score-Cutoff Wert", fmt("post_cutoff")),
            ("Reranking aktiv", bool_fmt("post_use_rerank")),
            ("Reranking-Modell", fmt("post_rerank_model")),
            ("Reranking top-n", fmt("post_rerank_top_n")),
            ("LongContextReorder", bool_fmt("post_use_reorder")),
        ]))

        # ── Query-Transformation ──────────────────────────────────────────────────
        groups.append(("Query-Transformation", [
            ("Query Rewrite", bool_fmt("use_query_rewrite")),
            ("HyDE", bool_fmt("use_hyde")),
            ("Query-Dekomposition", bool_fmt("use_query_decomposition")),
        ]))

        # ── Kontextverarbeitung ───────────────────────────────────────────────────
        groups.append(("Kontextverarbeitung", [
            ("Deduplizierung aktiv", bool_fmt("use_dedup")),
            ("Dedup-Schwellwert", fmt("dedup_threshold")),
            ("LLM-Konsolidierung", bool_fmt("use_llm_consolidation")),
        ]))

        # ── Generierung ───────────────────────────────────────────────────────────
        PROMPT_ALIASES = {
            "Use ONLY the context below to answer. If the answer is not explicitly stated":
                "PROMPT\\_STRICT",
            "Use ONLY the context below to answer. If the context describes a condition":
                "PROMPT\\_SOFT",
            "Answer the question using the context below as your primary source.":
                "PROMPT\\_PERMISSIVE",
        }

        prompt_text = prompt.get("pretext", "—")
        alias = None
        for key, a in PROMPT_ALIASES.items():
            if prompt_text.startswith(key):
                alias = a
                break
        if alias:
            prompt_display = alias
        else:
            prompt_display = prompt_text[:60] + "\\ldots" if len(prompt_text) > 60 else prompt_text
            prompt_display = prompt_display.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

        groups.append(("Generierung", [
            ("Response-Modus", fmt("response_mode")),
            ("Prompt-Pretext", prompt_display),
        ]))

        # ── Guards ────────────────────────────────────────────────────────────────
        groups.append(("Guards", [
            ("Confidence Guard aktiv", bool_fmt("use_confidence_guard")),
            ("Confidence Cutoff", fmt("confidence_cutoff")),
        ]))

        # ── Build LaTeX ───────────────────────────────────────────────────────────
        desc = phase_description or f"Konfiguration: {name}"

        def _build_table(grps, cap, lbl):
            t = []
            t.append("\\begin{table}[H]")
            t.append("\\centering")
            t.append("\\small")
            t.append("\\setlength{\\tabcolsep}{4pt}")
            t.append("\\begin{tabular}{lp{8cm}}")
            t.append("\\toprule")
            t.append("\\textbf{Parameter} & \\textbf{Wert} \\\\")
            for group_name, params in grps:
                t.append("\\midrule")
                t.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
                t.append("\\midrule")
                for param, value in params:
                    param_escaped = param.replace("_", "\\_")
                    t.append(f"\\quad {param_escaped} & {value} \\\\")
            t.append("\\bottomrule")
            t.append("\\end{tabular}")
            t.append(f"\\caption{{{cap}}}")
            t.append(f"\\label{{tab:{lbl}}}")
            t.append("\\end{table}")
            return "\n".join(t)

        mid = len(groups) // 2
        groups_a = groups[:mid]
        groups_b = groups[mid:]

        result_a = _build_table(
            groups_a,
            f"\\textbf{{{name}}} -- {desc} (1/2)",
            f"config_{name.replace('-', '_')}_a"
        )
        result_b = _build_table(
            groups_b,
            f"\\textbf{{{name}}} -- {desc} (2/2)",
            f"config_{name.replace('-', '_')}_b"
        )

        result = result_a + "\n\n" + result_b

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result)
        return result

    def export_latex_sample_questions_appendix(self, session_id: str,
                                               question_indices: list[int],
                                               save_path: str = None) -> str:
        """
        Generates a LaTeX appendix block for sample questions of a config.
        Per question: ID, question text, 4 eval scores, retrieved nodes preview.
        """
        config, queries = self.load_session(session_id)
        name = config.get("config_name", session_id)

        lines = []
        lines.append(f"\\subsubsection*{{\\texttt{{{name.replace('_', chr(92) + '_')}}}}}")
        lines.append(f"\\label{{app:samples:{name.replace('-', '_')}}}")
        lines.append("")

        for idx in question_indices:
            if idx >= len(queries):
                continue
            q = queries[idx]

            question = q.get("query", {}).get("original", "—")
            used_query = q.get("query", {}).get("used", question)
            evals = q.get("evaluation", {})
            nodes = q.get("retrieved_nodes", [])

            # eval scores
            def escore(metric):
                s = evals.get(metric, {}).get("score")
                return f"{s:.2f}" if s is not None else "—"

            corr = escore("correctness")
            faith = escore("faithfulness")
            relev = escore("relevancy")
            ctx = escore("context_relevancy")

            # answer (truncated)
            answer = str(q.get("answer", "")).strip().replace("\n", " ")
            if len(answer) > 300:
                answer = answer[:300] + "\\ldots"
            answer = answer.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")

            # question text escaped
            question_tex = question.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
            used_tex = used_query.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")

            lines.append(f"\\paragraph{{Q{idx}: {question_tex}}}")

            # rewritten query if different
            if used_query != question:
                lines.append(f"\\textit{{Umgeschrieben:}} {used_tex}\\\\[2pt]")

            # eval scores as inline colored boxes
            lines.append(
                f"\\noindent "
                f"\\textbf{{Corr.:}}~{corr}/5\\quad "
                f"\\textbf{{Faith.:}}~{faith}\\quad "
                f"\\textbf{{Relev.:}}~{relev}\\quad "
                f"\\textbf{{Ctx.Rel.:}}~{ctx}\\\\"
            )
            lines.append("")

            # answer
            lines.append(f"\\noindent\\textbf{{Antwort:}} {answer}")
            lines.append("")

            # retrieved nodes
            if nodes:
                lines.append("\\noindent\\textbf{Abgerufene Chunks:}")
                lines.append("\\begin{itemize}[noitemsep, topsep=2pt]")
                for node in nodes[:5]:  # show top 5
                    rank = node.get("rank", "?")
                    score = node.get("score")
                    score_s = f"{score:.4f}" if score is not None else "—"
                    cat = node.get("category") or "?"
                    topic = node.get("topic") or "?"
                    preview = str(node.get("preview", "")).replace("\n", " ").strip()
                    if len(preview) > 80:
                        preview = preview[:80] + "\\ldots"
                    preview = preview.replace("_", "\\_").replace("&", "\\&") \
                        .replace("%", "\\%").replace("#", "\\#") \
                        .replace("$", "\\$")
                    topic_tex = topic.replace("_", "\\_").replace("&", "\\&")
                    lines.append(
                        f"  \\item[\\textbf{{\\#{rank}}}] "
                        f"\\texttt{{[{score_s}]}} "
                        f"\\textit{{{cat}/{topic_tex}}} -- "
                        f"{preview}"
                    )
                lines.append("\\end{itemize}")

            lines.append("\\vspace{4pt}")
            lines.append("\\hrule")
            lines.append("\\vspace{6pt}")
            lines.append("")

        result = "\n".join(lines)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result)
        return result

# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Reporter")
    parser.add_argument("--log-dir", default="../logs", help="Path to logs directory")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all sessions")

    p_session = sub.add_parser("session", help="Display a single session")
    p_session.add_argument("session_id", help="Session folder name")
    p_session.add_argument("--no-queries",  action="store_true")
    p_session.add_argument("--show-nodes",  action="store_true")
    p_session.add_argument("--no-tokens",   action="store_true")
    p_session.add_argument("--no-eval",     action="store_true")

    sub.add_parser("compare", help="Compare all sessions side by side")

    args = parser.parse_args()
    reporter = Reporter(log_dir=args.log_dir)

    if args.cmd == "list":
        reporter.display_sessions_list()

    elif args.cmd == "session":
        reporter.display_session(
            args.session_id,
            show_queries=not args.no_queries,
            show_nodes=args.show_nodes,
            show_tokens=not args.no_tokens,
            show_eval=not args.no_eval,
        )

    elif args.cmd == "compare":
        reporter.display_comparison()

    else:
        # default: list + compare
        reporter.display_sessions_list()
        reporter.display_comparison()




if __name__ == "__main__":
    main()