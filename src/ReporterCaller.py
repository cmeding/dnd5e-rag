from Reporter import Reporter
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

def a_main():
    r = Reporter("../logs")
    # get all session IDs for phase 3 retrieval configs
    phase3 = [s for s in r.list_sessions() if
              "30-" in s or "31-" in s or "32-" in s or "33-" in s or "34-" in s or "35-" in s]
    print(r.export_latex_phase_table(phase3, caption="Phase 3: Retrieval Konfigurationen", label="phase3"))

    print(r.export_latex_sample_questions(
        phase3[0],
        question_indices=[0,1,2,4,5]
    ))

def b_main():
    r = Reporter("../logs")

    # generate all phase graphs and save as PDFs for LaTeX inclusion
    #phase2 = [s for s in r.list_sessions() if any(f"2{i}-" in s for i in range(7))]
    phase2 = r.get_sessions_by_phase_prefix(["20-", "21-", "22-", "23-", "24-", "25-", "26-", "27-"])
    r.plot_phase_comparison(phase2, metric="correctness",
                            title="Phase 2: Chunking — Avg Correctness",
                            save_path="../paper/img/phase2_correctness.pdf")

    r.plot_all_metrics_comparison(phase2,
                                  title="Phase 2: Chunking — Alle Metriken",
                                  save_path="../paper/img/phase2_all_metrics.pdf")

    r.plot_stacked_heatmap(
        [phase2[0]],
        title="Phase 2: Question Full Heatmap",
        save_path="../paper/img/phase2_question_full_heatmap.pdf",
        question_aliases=questions_aliases,
    )



    # LaTeX table
    print(r.export_latex_phase_table(phase2,
                                     caption="Phase 2: Chunking-Konfigurationen im Vergleich",

                                     label="phase2"))
def test_over_phase2():
    phase = r.get_sessions_by_phase_prefix([f"{2}{i}" for i in range(phase_ranges[2] + 1)])

    r.plot_phase_overview_heatmap_vertical(
        session_ids=phase,
        question_aliases=None,#{k: v for k, v in questions_aliases.items() if k < 15},
        question_filter=[i for i in range(0,30)],
        title="Heatmap Summary 1",
        save_path="../paper/img/phase2_v_summary.pdf",
        show_cell_text=True,
        config_label="prefix",
        config_filter=[i for i in range(0,8)],
        axis_font_scale=0.30,
        cell_font_scale=0.25,
    )

    """
    r.plot_phase_question_metric_heatmap_vertical(
        session_ids=phase,
        question_aliases=None,#{k: v for k, v in questions_aliases.items() if k < 15},
        question_filter=[i for i in range(0,16)],
        title="Heatmap Metrics 1",
        save_path="../paper/img/phase2_v_metrics.pdf",
    )

    r.plot_phase_overview_heatmap(
        session_ids=phase,
        question_aliases=None,  # {k: v for k, v in questions_aliases.items() if k < 15},
        question_filter=[i for i in range(0, 16)],
        title="Heatmap Summary 1",
        save_path="../paper/img/phase2_h_summary.pdf",
        show_cell_text=False,
    )

    r.plot_phase_question_metric_heatmap(
        session_ids=phase,
        question_aliases={k: v for k, v in questions_aliases.items() if k < 15},
        title="Heatmap Metrics 1",
        save_path="../paper/img/phase2_h_metrics.pdf",
    )
    
    """

def test2_over_phase2():
    phase = r.get_sessions_by_phase_prefix([f"{2}{i}" for i in range(phase_ranges[2] + 1)])

    r.plot_phase_bar_chart_horizontal(
        session_ids=phase,
        config_label="name",
        title="Phase 2 Bar Chart",
        save_path="../paper/img/phase2_bar_chart_v.pdf",
        axis_font_scale=0.25,
        bar_label_scale=0.15,
    )

    r.export_latex_eval_table(
        session_ids=phase,
        caption="Phase 2 Eval Table",
        label="phase2-eval-table",
        save_path="../paper/tex/phase2_eval_table.tex",
    )

    r.export_latex_cost_table(
        session_ids=phase,
        caption="Phase 2 Cost Table",
        label="phase2-cost-table",
        save_path="../paper/tex/phase2_cost_table.tex",
    )

def make_index_time_stuff():
    r = Reporter("../logs")
    r.plot_index_build_stats(
        title="Phase 2: Chunking — Index Build-Statistiken",
        save_path="../paper/img/phase2_index_build.pdf",
        bar_label_scale=0.22,
        axis_font_scale=0.3
    )

def main():
    pass

if __name__ == "__main__":
    phase_loop()