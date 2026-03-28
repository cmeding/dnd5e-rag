from src.old.Reporter_old import Reporter
r = Reporter("../logs")


phase_ranges = {
    1: 0,
    2: 7,
    3: 5,
    4: 3,
    5: 2,
    6: 1,
    7: 2,
    8: 1,
    9: 3
}
phase_names = {
    1:  "Baseline (kein RAG)",
    2:  "Indexierung",
    3:  "Retrieval",
    4:  "Postprocessing",
    5:  "Query-Transformation",
    6:  "Kontextverarbeitung",
    7:  "Prompt-Template",
    8:  "Response-Modus",
    9:  "Kombinationen",
}
questions_aliases = {
    0:  "Fireball Range",
    1:  "Aboleth AC",
    2:  "Inflict Wounds Damage",
    3:  "Haste Duration",
    4:  "Fighter Hit Die",
    5:  "Grappled + Spells",
    6:  "Frightened + Dash",
    7:  "Restrained + Attacks",
    8:  "Invisible + Spells",
    9:  "Charmed + Non-Charmers",
    10: "Concentration + Unconscious",
    11: "Sanctuary + Smite",
    12: "Exhaustion 3 + Restrained",
    13: "Paralyzed + Death Saves",
    14: "Grapple Ends When",
    15: "Nat 20 = Max Damage?",
    16: "Action Surge + 2 Spells",
    17: "Sneak Attack + Hidden?",
    18: "Proficiency + All Saves?",
    19: "Druid + Metal Armor",
    20: "Wish Casting Time",
    21: "Giant Eagle Speed",
    22: "Human Languages",
    23: "Deafened + Verbal",
    24: "Stunned + Saves",
    25: "Incapacitated + Reactions",
    26: "Frightened ≠ Charmed Immune",
    27: "Restrained + Hover",
    28: "Barbarian Rage + Spells",
    29: "Spell + Attack Same Turn",
}
show_question_ids = [3, 5, 12, 16]

def phase_loop():
    import os
    for d in ["../paper/img/results", "../paper/tex/results", "../paper/tex/figures"]:
        os.makedirs(d, exist_ok=True)

    for k, v in phase_ranges.items():
        phase = r.get_sessions_by_phase_prefix([f"{k}{i}-" for i in range(v + 1)])
        if not phase:
            print(f"Skipping phase {k} — no sessions found")
            continue

        is_single = len(phase) == 1
        phase_name = phase_names[k]
        title_prefix = f"Phase {k}: {phase_name}"

        titles = {
            "bar":        f"{title_prefix} -- Ø Evaluationsmetriken pro Konfiguration",
            "eval_table": f"{title_prefix} -- Durchschnittliche Evaluationsmetriken",
            "cost_table": f"{title_prefix} -- Durchschnittliche Token-, Kosten- und Zeitwerte",
            "overview":   f"{title_prefix} -- Config × Frage Übersicht",
            "metrics_a":  f"{title_prefix} -- Metrik × Frage, Q0--Q14 (Ø über alle Configs)",
            "metrics_b":  f"{title_prefix} -- Metrik × Frage, Q15--Q29 (Ø über alle Configs)",
        }
        labels = {
            "bar":        f"phase{k}_bar",
            "eval_table": f"phase{k}_eval",
            "cost_table": f"phase{k}_cost",
            "overview":   f"phase{k}_overview",
            "metrics_a":  f"phase{k}_metrics_a",
            "metrics_b":  f"phase{k}_metrics_b",
        }
        paths = {
            "bar":        f"../paper/img/results/phase{k}_bar.pdf",
            "eval_table": f"../paper/tex/results/phase{k}_eval_table.tex",
            "cost_table": f"../paper/tex/results/phase{k}_cost_table.tex",
            "overview":   f"../paper/img/results/phase{k}_overview.pdf",
            "metrics_a":  f"../paper/img/results/phase{k}_metrics_a.pdf",
            "metrics_b":  f"../paper/img/results/phase{k}_metrics_b.pdf",
        }
        fig_paths = {
            "bar":      f"../paper/tex/figures/phase{k}_bar.tex",
            "overview": f"../paper/tex/figures/phase{k}_overview.tex",
            "metrics":  f"../paper/tex/figures/phase{k}_metrics.tex",
        }

        # always generate these
        r.plot_phase_bar_chart_horizontal(
            session_ids=phase, config_label="name",
            title=titles["bar"], save_path=paths["bar"],
            axis_font_scale=0.25, bar_label_scale=0.15,
        )
        r.export_latex_eval_table(
            session_ids=phase, caption=titles["eval_table"],
            label=labels["eval_table"], save_path=paths["eval_table"],
        )
        r.export_latex_cost_table(
            session_ids=phase, caption=titles["cost_table"],
            label=labels["cost_table"], save_path=paths["cost_table"],
        )
        r.export_latex_figure_bar(
            img_path=f"img/results/phase{k}_bar.pdf",
            caption=titles["bar"], label=labels["bar"],
            save_path=fig_paths["bar"],
        )

        # skip heatmaps for single-session phases
        if is_single:
            print(f"Phase {k}: single session — skipping heatmaps")
            continue

        r.plot_phase_overview_heatmap_vertical(
            session_ids=phase, question_aliases=None,
            question_filter=list(range(0, 30)),
            title=titles["overview"], save_path=paths["overview"],
            show_cell_text=True, config_label="prefix",
            config_filter=list(range(0, v + 1)),
            axis_font_scale=0.30, cell_font_scale=0.25,
        )
        r.plot_phase_question_metric_heatmap_vertical(
            session_ids=phase, question_aliases=None,
            question_filter=list(range(0, 15)),
            title=titles["metrics_a"], save_path=paths["metrics_a"],
        )
        r.plot_phase_question_metric_heatmap_vertical(
            session_ids=phase, question_aliases=None,
            question_filter=list(range(15, 30)),
            title=titles["metrics_b"], save_path=paths["metrics_b"],
        )
        r.export_latex_figure_overview(
            img_path=f"img/results/phase{k}_overview.pdf",
            caption=titles["overview"], label=labels["overview"],
            save_path=fig_paths["overview"],
        )
        r.export_latex_figure_metrics(
            img_path_a=f"img/results/phase{k}_metrics_a.pdf",
            img_path_b=f"img/results/phase{k}_metrics_b.pdf",
            caption=f"{title_prefix} -- Metrik × Frage (Ø über alle Konfigurationen)",
            label=f"phase{k}_metrics",
            save_path=fig_paths["metrics"],
        )

def make_index_times():
    r = Reporter("../logs")
    r.plot_index_build_stats(
        title="Phase 2: Chunking — Index Build-Statistiken",
        save_path="../paper/img/phase2_index_build.pdf",
        bar_label_scale=0.22,
        axis_font_scale=0.3
    )

def final_gens():
    r = Reporter("../logs")
    all_sessions = r.get_sessions_by_phase_prefix(
        [f"{k}{i}-" for k, v in phase_ranges.items() for i in range(v + 1)]
    )

    # one boxplot per metric
    for metric in ["correctness", "faithfulness", "relevancy", "context_relevancy"]:
        r.plot_metric_boxplot(
            all_sessions, metric=metric,
            config_label="prefix",
            title=f"Gesamtergebnis — {metric.capitalize()} alle Konfigurationen",
            save_path=f"../paper/img/results/overall_boxplot_{metric}.pdf",
            axis_font_scale=0.45,
        )

    # ranking table
    r.export_latex_ranking_table(
        all_sessions,
        sort_by="correctness",
        caption="Gesamtranking aller Konfigurationen nach durchschnittlicher Correctness",
        label="overall_ranking",
        save_path="../paper/tex/results/overall_ranking.tex",
    )

config_descriptions = {
    "10-baseline-no-rag":           "Baseline -- Direktes LLM ohne RAG-Retrieval",
    "20-index-token-default":       "Indexierung -- Token-Splitting ohne Markdown-Vorparser",
    "21-index-token-markdown":      "Indexierung -- Token-Splitting mit Markdown-Vorparser",
    "22-index-sentences-default":   "Indexierung -- Sentence-Splitting ohne Markdown-Vorparser",
    "23-index-sentences-markdown":  "Indexierung -- Sentence-Splitting mit Markdown-Vorparser (Basisindex)",
    "24-index-sentwindow-default":  "Indexierung -- Sentence-Window-Verfahren ohne Markdown-Vorparser",
    "25-index-sentwindow-markdown": "Indexierung -- Sentence-Window-Verfahren mit Markdown-Vorparser",
    "26-index-hierarchical-default":  "Indexierung -- Hierarchisches Chunking ohne Markdown-Vorparser",
    "27-index-hierarchical-markdown": "Indexierung -- Hierarchisches Chunking mit Markdown-Vorparser",
    "30-retrieval-bm25":            "Retrieval -- BM25 Hybrid-Retrieval (Keyword + Vektor)",
    "31-retrieval-fusion-2q":       "Retrieval -- Query-Fusion mit 2 Anfragevarianten",
    "32-retrieval-fusion-3q":       "Retrieval -- Query-Fusion mit 3 Anfragevarianten",
    "33-retrieval-bm25-fusion":     "Retrieval -- BM25 kombiniert mit Query-Fusion (2 Varianten)",
    "34-retrieval-topk5":           "Retrieval -- Reduziertes top-k=5",
    "35-retrieval-topk15":          "Retrieval -- Erweitertes top-k=15",
    "40-post-rerank":               "Postprocessing -- Cross-Encoder Reranking (top-n=5)",
    "41-post-reorder":              "Postprocessing -- LongContextReorder (Lost-in-the-Middle)",
    "42-post-rerank-reorder":       "Postprocessing -- Reranking kombiniert mit Reorder",
    "43-post-cutoff":               "Postprocessing -- Score-Cutoff bei 0.015 (RRF-Schwellwert)",
    "50-transform-rewrite":         "Query-Transformation -- Query-Rewrite via Meta-LLM",
    "51-transform-hyde":            "Query-Transformation -- HyDE (Hypothetical Document Embedding)",
    "52-transform-decomposition":   "Query-Transformation -- Query-Dekomposition in Teilfragen",
    "60-context-dedup":             "Kontextverarbeitung -- Jaccard-Deduplizierung (Schwellwert 0.80)",
    "61-context-consolidation":     "Kontextverarbeitung -- LLM-basierte Kontextkomprimierung",
    "70-prompt-strict":             "Prompt-Template -- Striktes Template (nur explizites Regelwissen)",
    "71-prompt-soft":               "Prompt-Template -- Weiches Template (Schlussfolgerungen erlaubt)",
    "72-prompt-permissive":         "Prompt-Template -- Permissives Template (freies Schlussfolgern)",
    "80-mode-refine":               "Response-Modus -- REFINE (iterative Chunk-Verfeinerung)",
    "81-mode-tree-summarize":       "Response-Modus -- TREE\\_SUMMARIZE (hierarchische Zusammenfassung)",
    "90-combined-best-guess":       "Kombination -- Beste Einzelmodule kombiniert (Sentences-Markdown-Index)",
    "91-combined-sentwindow-best":  "Kombination -- Beste Module mit Sentence-Window-Index",
    "92-combined-best-permissive":  "Kombination -- Beste Module mit permissivem Prompt-Template",
    "93-kitchen-sink":              "Kombination -- Alle verfügbaren Module aktiviert (Kitchen-Sink)",
}

def config_tables():
    all_sessions = r.get_sessions_by_phase_prefix(
        [f"{k}{i}-" for k, v in phase_ranges.items() for i in range(v + 1)]
    )
    for sid in all_sessions:
        config, _ = r.load_session(sid)
        name = config.get("config_name", sid)
        description = config_descriptions.get(name, f"Konfiguration: {name}")
        r.export_latex_config_table(
            sid,
            phase_description=description,
            save_path=f"../paper/tex/configs/{name}_params.tex"
        )

def questions():
    show_question_ids = [3, 5, 12, 16]

    all_sessions = r.get_sessions_by_phase_prefix(
        [f"{k}{i}-" for k, v in phase_ranges.items() for i in range(v + 1)]
    )

    for sid in all_sessions:
        config, _ = r.load_session(sid)
        name = config.get("config_name", sid)
        r.export_latex_sample_questions_appendix(
            sid,
            question_indices=show_question_ids,
            save_path=f"../paper/tex/samples/{name}_samples.tex"
        )

if __name__ == "__main__":
    pass