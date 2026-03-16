from RAGv4 import RAGPipeline, key_from_file
from loguru import logger
import time

KEYFILE = "./key.txt"
BASE_QUESTION = "Can a grappled Creature cast Spells?"

MINIMUM_RAG = {
    "ind_name":              "SRD-Minimum-v1",
    "embed_split_on":        "tokens",
    "special_mode":          "none",
    "retriever_top_k":       5,
    "retriever_query_variants": 1,
    "retriever_with_keywords": False,
    "post_use_cutoff":       False,
    "post_use_rerank":       False,
    "post_use_reorder":      False,
    "use_hyde":              False,
    "use_query_rewrite":     False,
    "use_llm_metadata_filter": False,
    "use_confidence_guard":  False,
    "use_query_decomposition": False,
    "use_dedup":             False,
    "use_llm_consolidation": False,
    "eval_faithfulness":     False,
    "eval_relevancy":        False,
    "eval_correctness":      False,
}

MAXIMUM_RAG = {
    "ind_name":              "SRD-Maximum-v1",
    "embed_with_markdown":   True,
    "embed_split_on":        "sentences",
    "special_mode":          "sentence_window",
    "retriever_top_k":       15,
    "retriever_query_variants": 4,
    "retriever_with_keywords": True,
    "post_use_cutoff":       False,
    "post_cutoff":           0.3,
    "post_use_rerank":       True,
    "post_rerank_top_n":     5,
    "post_use_reorder":      True,
    "use_hyde":              True,
    "use_query_rewrite":     True,
    "use_confidence_guard":  True,
    "use_query_decomposition": True,
    "use_dedup":             True,
    "use_llm_consolidation": True,
    "eval_faithfulness":     True,
    "eval_relevancy":        True,
    "eval_correctness":      True,
    "eval_reference": (
        "Yes. The Grappled condition only reduces speed to 0. "
        "It does not restrict spellcasting in any way."
    ),
}

TEST_RAG = {
    "ind_name": "SRD-Test-v4",
    "eval_faithfulness": True,
    "eval_relevancy": True,
    "eval_correctness": True,
    "eval_context_relevancy": True,
    "eval_reference": (
        "Yes. The Grappled condition only reduces speed to 0. "
        "It does not restrict spellcasting or other actions in any way."
    ),
    "verbose": True,
    "log_queries": True,
    "config_name": "DEFAULT-EVAL",
    "session_name": "Context-Relevancy-01",
}


INDEX_SETTING_COMBINATIONS = [
    {
        "ind_name": "token-default",
        "config_name": "BASE-token-default",
        "embed_split_on": "tokens",
        "embed_with_markdown": False,
        "special_mode": "none",
    },
    {
        "ind_name": "token-markdown",
        "config_name": "BASE-token-markdown",
        "embed_split_on": "tokens",
        "embed_with_markdown": True,
        "special_mode": "none",
    },
    {
        "ind_name": "sentences-default",
        "config_name": "BASE-sentences-default",
        "embed_split_on": "sentences",
        "embed_with_markdown": False,
        "special_mode": "none",
    },
    {
        "ind_name": "sentences-markdown",
        "config_name": "BASE-sentences-markdown",
        "embed_split_on": "sentences",
        "embed_with_markdown": True,
        "special_mode": "none",
    },
    {
        "ind_name": "sentence_window-default",
        "config_name": "BASE-sentence_window-default",
        "embed_with_markdown": False,
        "special_mode": "sentence_window",
    },
    {
        "ind_name": "sentence_window-markdown",
        "config_name": "BASE-sentence_window-markdown",
        "embed_with_markdown": True,
        "special_mode": "sentence_window",
    },
    {
        "ind_name": "hierarchical-default",
        "config_name": "BASE-hierarchical-default",
        "embed_with_markdown": False,
        "special_mode": "hierarchical",
    },
    {
        "ind_name": "hierarchical-markdown",
        "config_name": "BASE-hierarchical-markdown",
        "embed_with_markdown": True,
        "special_mode": "hierarchical",
    }

]

def test_rag():
    key = key_from_file(KEYFILE)

    print("\n" + "=" * 60)
    print("  RUNTIME TEST — TEST RAG")
    print("=" * 60)
    min_rag = RAGPipeline(APIKey=key, **TEST_RAG)
    min_rag.RAG_query(BASE_QUESTION)

    print("\n" + "=" * 60)
    print("Finished TEST RAG")

def runtime_test():
    key = key_from_file(KEYFILE)

    print("\n" + "=" * 60)
    print("  RUNTIME TEST — DIRECT LLM (no RAG)")
    print("=" * 60)
    baseline = RAGPipeline(APIKey=key, do_startup_setup=True)
    baseline.simple_query(BASE_QUESTION)

    print("\n" + "=" * 60)
    print("  RUNTIME TEST — MINIMUM RAG")
    print("=" * 60)
    min_rag = RAGPipeline(APIKey=key, **MINIMUM_RAG)
    min_rag.RAG_query(BASE_QUESTION)


    print("\n" + "=" * 60)
    print("  RUNTIME TEST — MAXIMUM RAG")
    print("=" * 60)
    max_rag = RAGPipeline(APIKey=key, **MAXIMUM_RAG)
    max_rag.RAG_query(BASE_QUESTION)

    print("\n" + "=" * 60)
    print("  RUNTIME TEST COMPLETE")
    print("=" * 60)

def index_run():
    key = key_from_file(KEYFILE)
    for i_rag in INDEX_SETTING_COMBINATIONS:
        if not (i_rag["ind_name"] == "hierarchical-default" or i_rag["ind_name"] == "hierarchical-markdown"):
            continue
        print("\n" + "=" * 60)
        print(i_rag["ind_name"])
        print("\n" + "=" * 60)
        rag = RAGPipeline(APIKey=key, **i_rag, **EVAL_SETTINGS, session_name="index-build",)
        rag.RAG_query(BASE_QUESTION)
        print("\n" + "=" * 60)
        print("\n" + "=" * 60)
    print("FINISHED WITH INDEX BUILD")

def single_test(config):
    key = key_from_file(KEYFILE)
    prompt_pretext = (
        "Use ONLY the context below to answer. "
        "If the context describes a condition or rule, you may reason about "
        "what it does and does not restrict. "
        "If the answer cannot be determined from the context at all, "
        "respond with: 'I cannot find this in the provided rules.' "
        "Do not use outside knowledge beyond what the context implies."
    )
    print("\n" + "=" * 60)
    print(config["ind_name"])
    print("\n" + "=" * 60)
    rag = RAGPipeline(APIKey=key, **config, **EVAL_SETTINGS, prompt_pretext=prompt_pretext, session_name="BT0")
    #rag.RAG_query(BASE_QUESTION)
    rag.simple_query(BASE_QUESTION)
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)

def batch_test(config, questions, ground_truths):
    key = key_from_file(KEYFILE)
    prompt_pretext = (
        "Use ONLY the context below to answer. "
        "If the context describes a condition or rule, you may reason about "
        "what it does and does not restrict. "
        "If the answer cannot be determined from the context at all, "
        "respond with: 'I cannot find this in the provided rules.' "
        "Do not use outside knowledge beyond what the context implies."
    )

    print("\n" + "=" * 60)
    print(config["ind_name"])
    print("\n" + "=" * 60)
    rag = RAGPipeline(APIKey=key, **config, **EVAL_SETTINGS, prompt_pretext=prompt_pretext, session_name="20q-test-v1")
    # rag.RAG_query(BASE_QUESTION)
    #rag.simple_query(BASE_QUESTION)
    rag.batch_simple_query(queries=questions, references=ground_truths)
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)

BEST_GUESS = {
    # Indexing — sentence splitter with markdown awareness
    # SRD files are well-structured markdown, so the parser can use that structure
    # sentence splitting respects natural rule boundaries better than token splitting
    "ind_name":            "sentences-markdown",
    "config_name": "BEST-G-sent-md-v2",
    "special_mode":        "none",
    "embed_split_on":      "sentences",
    "embed_with_markdown": True,
    "chunk_size":          512,
    "chunk_overlay":       50,
    "generate_metadata":   True,

    # Retrieval — hybrid vector + BM25
    # DnD rules have very specific terminology (grappled, concentration, spell slots)
    # BM25 is strong on exact keyword matches which matters a lot here
    "retriever_top_k":           10,
    "retriever_with_keywords":   True,
    "retriever_query_variants":  2,

    # Postprocessing — rerank then reorder
    # cross-encoder reranking is the single highest-impact postprocessing step
    # reorder puts best chunks at context edges where LLMs attend better
    "post_use_rerank":   True,
    "post_rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "post_rerank_top_n": 5,
    "post_use_reorder":  True,
    "post_use_cutoff":   False,  # cutoff after rerank is redundant

    # Query transformation — rewrite only
    # rewrite helps with vague natural language questions
    # HyDE tends to hallucinate DnD rules it doesn't know
    # decomposition overkill for single-fact rule lookups
    "use_query_rewrite":      True,
    "use_hyde":               False,
    "use_query_decomposition":False,

    # Context — dedup only, skip consolidation
    # dedup removes redundant chunks that often appear in the SRD
    # consolidation adds latency and meta LLM cost for questionable gain
    "use_dedup":             True,
    "dedup_threshold":       0.80,
    "use_llm_consolidation": False,

    # Response
    "response_mode": "COMPACT",
}

OLD_GROUND_TRUTH = [
    # Type A
    "Fireball has a range of 150 feet.",
    "An Aboleth has an Armor Class of 17.",
    "Inflict Wounds deals necrotic damage.",
    "Haste lasts up to 1 minute and requires concentration.",
    "A Fighter uses a d10 as their hit die.",

    # Type B
    "Yes. The grappled condition only reduces the creature's speed to 0. It imposes no restriction on spellcasting.",
    "Yes. The frightened condition only imposes disadvantage on ability checks and attack rolls while the source of fear is in sight, and prevents moving closer to the source. It does not restrict which actions a creature can take.",
    "Yes. The restrained condition does not prevent attack rolls, though attack rolls made by a restrained creature have disadvantage and attack rolls against it have advantage.",
    "No. The invisible condition grants advantage on attack rolls and imposes disadvantage on attack rolls against the creature. It does not restrict spellcasting.",
    "No. The charmed condition only prevents the charmed creature from attacking the charmer and from targeting them with harmful abilities or magical effects. It places no restriction on actions against other creatures.",

    # Type C
    "Concentration ends immediately when a caster falls unconscious, ending any concentration spell that was active.",
    "Yes. Divine Smite is a class feature, not a spell. Sanctuary ends only when the protected creature casts a spell that affects an enemy or takes an action that targets an enemy with a harmful effect. Using Divine Smite does not end Sanctuary.",
    "Exhaustion level 3 halves movement speed. The restrained condition reduces speed to 0. Speed of 0 takes precedence — the creature cannot move at all.",
    "No. A paralyzed creature is incapacitated. Incapacitated creatures cannot take actions, and making a death saving throw requires an action. Therefore a paralyzed creature cannot make death saving throws.",
    "The grappled condition ends if the grappler is incapacitated, or if an effect removes the grappled creature from the reach of the grappler or the grappling effect.",

    # Type D
    "No. A critical hit on a natural 20 doubles the number of damage dice rolled. It does not automatically deal maximum damage.",
    "Yes. Action Surge grants an additional action. The restriction on casting two leveled spells applies only when one of them is cast using a bonus action. Casting a leveled spell with a normal action and another leveled spell with an Action Surge action is permitted.",
    "No. Sneak Attack requires either advantage on the attack roll, or that an ally of the rogue is adjacent to the target and that ally is not incapacitated. Being hidden is not required.",
    "No. Proficiency bonus applies only to saving throws for which the creature has proficiency, as determined by its class or other features.",
    "Druids will not wear metal armor by tradition — the class description states they will not — but there is no mechanical rule preventing it. A Druid who wears metal armor violates the class ethos but suffers no mechanical penalty from the rules as written.",

]

# -------------------------------------------------------------------------------

QUESTIONS = [
    # Type A — Explicit single-fact lookup
    "What is the range of the Fireball spell?",
    "What is the Armor Class of an Aboleth?",
    "What damage type does Inflict Wounds deal?",
    "How long does the Haste spell last?",
    "How many hit dice does a Fighter have?",

    # Type B — Answer by omission (correct answer = absence of restriction)
    "Can a grappled creature cast spells?",
    "Can a frightened creature take the Dash action?",
    "Can a restrained creature still make attack rolls?",
    "Does the invisible condition prevent a creature from casting spells?",
    "Does being charmed prevent a creature from attacking non-charmers?",

    # Type C — Multi-chunk / combined reasoning
    "What happens to concentration spells when a caster falls unconscious?",
    "Can a Paladin under the Sanctuary spell use Divine Smite?",
    "If a creature has exhaustion level 3 and is also restrained, what is its movement speed?",
    "Can a paralyzed creature make death saving throws?",
    "What conditions cause the Grappled condition to end?",

    # Type D — Common misconceptions (LLM likely knows the wrong answer)
    "Does a natural 20 on an attack roll automatically deal maximum damage?",
    "Can you use Action Surge to cast two leveled spells in one turn?",
    "Does Sneak Attack require the rogue to be hidden?",
    "Do you add your proficiency bonus to every saving throw?",
    "Can a Druid wear metal armor?",

# Type A — Explicit single-fact lookup
    "What is the casting time of the Wish spell?",
    "What is the movement speed of a Giant Eagle?",
    "How many languages does a standard Human character speak at character creation?",

    # Type B — Answer by omission
    "Can a deafened creature cast spells with verbal components?",
    "Does the stunned condition prevent a creature from making saving throws?",
    "Can an incapacitated creature take reactions?",

    # Type C — Multi-chunk / combined reasoning
    "If a creature is immune to the frightened condition, can it still be charmed?",
    "Can a creature with a fly speed of 0 feet due to the restrained condition still hover?",

    # Type D — Common misconceptions
    "Does a Barbarian lose their Rage bonus damage if they cast a spell while raging?",
    "Can you cast a spell and also make a weapon attack in the same turn without any special features?",
]

GROUND_TRUTH = [
    # Type A — Explicit single-fact lookup
    "Fireball has a range of 150 feet, with a point of origin within that range. "
    "It creates a 20-foot radius sphere of fire at that point.",

    "An Aboleth has an Armor Class of 17 (natural armor).",

    "Inflict Wounds deals necrotic damage.",

    "Haste lasts up to 1 minute and requires concentration.",

    "A Fighter uses a d10 as their hit die.",

    # Type B — Answer by omission
    "Yes. The grappled condition only reduces the creature's speed to 0. "
    "It imposes no restriction on spellcasting.",

    "Yes. The frightened condition only imposes disadvantage on ability checks "
    "and attack rolls while the source of fear is in sight, and prevents moving "
    "closer to the source. It does not restrict which actions a creature can take.",

    "Yes. The restrained condition does not prevent attack rolls, though attack "
    "rolls made by a restrained creature have disadvantage and attack rolls "
    "against it have advantage.",

    "No. The invisible condition grants advantage on attack rolls and imposes "
    "disadvantage on attack rolls against the creature. "
    "It does not restrict spellcasting.",

    "No. The charmed condition only prevents the charmed creature from attacking "
    "the charmer and from targeting them with harmful abilities or magical effects. "
    "It places no restriction on actions against other creatures.",

    # Type C — Multi-chunk / combined reasoning
    "Concentration ends immediately when a caster falls unconscious, "
    "ending any concentration spell that was active.",

    "No. While Divine Smite is a class feature rather than a spell, it deals "
    "damage as part of an attack. Sanctuary ends when the protected creature "
    "attacks any other creature or casts a spell that affects an enemy. "
    "Using Divine Smite as part of an attack against an enemy ends the Sanctuary spell.",

    "Exhaustion level 3 halves movement speed. The restrained condition reduces "
    "speed to 0. A speed of 0 takes precedence — the creature cannot move at all.",

    "Yes. A paralyzed creature is incapacitated and automatically fails Strength "
    "and Dexterity saving throws. However, death saving throws are not Strength "
    "or Dexterity saves, and making a death saving throw is not an action — it "
    "happens automatically at the start of the creature's turn. "
    "A paralyzed creature therefore still makes death saving throws normally.",

    "The grappled condition ends if the grappler is incapacitated, or if an effect "
    "removes the grappled creature from the reach of the grappler or the grappling effect.",

    # Type D — Common misconceptions
    "No. A critical hit on a natural 20 doubles the number of damage dice rolled. "
    "It does not automatically deal maximum damage.",

    "Yes. Action Surge grants an additional action. The restriction on casting two "
    "leveled spells applies only when one of them is cast using a bonus action. "
    "Casting a leveled spell with a normal action and another leveled spell with "
    "an Action Surge action is permitted.",

    "No. Sneak Attack requires either advantage on the attack roll, or that an ally "
    "of the rogue is adjacent to the target and that ally is not incapacitated. "
    "Being hidden is not required.",

    "No. Proficiency bonus applies only to saving throws for which the creature "
    "has proficiency, as determined by its class or other features.",

    "Druids will not wear metal armor by tradition — the class description states "
    "they will not — but there is no mechanical rule preventing it. A Druid who "
    "wears metal armor violates the class ethos but suffers no mechanical penalty "
    "from the rules as written.",

# Type A
    "The casting time of the Wish spell is 1 action.",

    "A Giant Eagle has a walking speed of 10 feet and a flying speed of 80 feet.",

    "A standard Human speaks Common and one additional language of the player's choice.",

    # Type B
    "Yes. The deafened condition only prevents a creature from hearing and causes "
    "it to automatically fail any ability check that requires hearing. "
    "It does not prevent the use of verbal spell components — the rules do not "
    "require a caster to hear themselves to cast a spell with a verbal component.",

    "No. The stunned condition does not prevent saving throws. A stunned creature "
    "is incapacitated, cannot move, can only speak falteringly, automatically fails "
    "Strength and Dexterity saving throws, and attack rolls against it have advantage. "
    "Saving throws of other types are unaffected.",

    "No. An incapacitated creature cannot take actions or reactions.",

    # Type C
    "Yes. Immunity to frightened and immunity to charmed are separate conditions. "
    "A creature immune to one is not automatically immune to the other unless "
    "explicitly stated in its stat block or traits.",

    "A restrained creature has its speed reduced to 0, which prevents it from moving. "
    "However, the rules on hovering state that a creature with a fly speed that is "
    "reduced to 0 while airborne falls unless it can hover. Whether the creature "
    "falls depends on whether it has the hover trait, not on the restrained condition itself.",

    # Type D
    "A Barbarian cannot cast spells or concentrate on spells while raging. "
    "This is an absolute restriction — spellcasting is not possible during rage. "
    "To cast spells, the Barbarian must wait until the rage has ended or end it voluntarily beforehand.",

    "No. Casting a spell typically uses your action. Making a weapon attack also uses "
    "your action. You cannot do both in the same turn unless you have a feature that "
    "specifically grants an additional attack or action, such as the Fighter's Action "
    "Surge, or a spell that grants a bonus action attack.",

]

EVAL_SETTINGS = {
    "eval_faithfulness": True,
    "eval_relevancy": True,
    "eval_correctness": True,
    "eval_context_relevancy": True,
    "eval_reference": (
        "See Reference"
    ),
    "verbose": True,
    "log_queries": True,
}

PROMPT_STRICT = (
    "Use ONLY the context below to answer. "
    "If the answer is not explicitly stated in the context, "
    "respond with exactly: 'I cannot find this in the provided rules.' "
    "Do not infer, guess, or use outside knowledge."
)
PROMPT_SOFT = (
    "Use ONLY the context below to answer. "
    "If the context describes a condition or rule, you may reason about "
    "what it does and does not restrict. "
    "If the answer cannot be determined from the context at all, "
    "respond with: 'I cannot find this in the provided rules.' "
    "Do not use outside knowledge beyond what the context implies."
)
PROMPT_PERMISSIVE = (
    "Answer the question using the context below as your primary source. "
    "You may use reasoning and inference based on the rules described. "
    "If the context contains no relevant information at all, say so."
)

BASE = {
    "kb_source":               "../data/DND.SRD.Wiki-0.5.2",
    "ind_dir":                 "../index",
    "rag_llm_model":           "mistralai/mistral-7b-instruct-v0.1",
    "meta_llm_model":          "openai/gpt-4o-mini",
    "meta_temperature":        0.0,
    "llm_temperature":         0.1,
    "llm_max_response":        512,
    "retriever_top_k":         10,
    "retriever_query_variants":1,
    "retriever_with_keywords": False,
    "post_use_cutoff":         False,
    "post_use_rerank":         False,
    "post_use_reorder":        False,
    "use_hyde":                False,
    "use_query_rewrite":       False,
    "use_query_decomposition": False,
    "use_dedup":               False,
    "use_llm_consolidation":   False,
    "use_confidence_guard":    False,
    "prompt_pretext":          PROMPT_SOFT,
    "response_mode":           "COMPACT",
}

CONFIGS = [
    # ── Phase 1: Baseline (simple_only) ──────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "10-baseline-no-rag", "_simple_only": True},

    # ── Phase 2: Index comparison ─────────────────────────────────────────────
    {**BASE, "ind_name": "token-default",           "config_name": "20-index-token-default"},
    {**BASE, "ind_name": "token-markdown",          "config_name": "21-index-token-markdown"},
    {**BASE, "ind_name": "sentences-default",       "config_name": "22-index-sentences-default"},
    {**BASE, "ind_name": "sentences-markdown",      "config_name": "23-index-sentences-markdown"},
    {**BASE, "ind_name": "sentence_window-default", "config_name": "24-index-sentwindow-default"},
    {**BASE, "ind_name": "sentence_window-markdown","config_name": "25-index-sentwindow-markdown"},
    {**BASE, "ind_name": "hierarchical-default",    "config_name": "26-index-hierarchical-default"},
    {**BASE, "ind_name": "hierarchical-markdown",   "config_name": "27-index-hierarchical-markdown"},

    # ── Phase 3: Retrieval ────────────────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "30-retrieval-bm25",         "retriever_with_keywords": True},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "31-retrieval-fusion-2q",    "retriever_query_variants": 2},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "32-retrieval-fusion-3q",    "retriever_query_variants": 3},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "33-retrieval-bm25-fusion",  "retriever_with_keywords": True, "retriever_query_variants": 2},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "34-retrieval-topk5",        "retriever_top_k": 5},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "35-retrieval-topk15",       "retriever_top_k": 15},

    # ── Phase 4: Postprocessing ───────────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "40-post-rerank",         "post_use_rerank": True, "post_rerank_top_n": 5},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "41-post-reorder",        "post_use_reorder": True},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "42-post-rerank-reorder", "post_use_rerank": True, "post_rerank_top_n": 5, "post_use_reorder": True},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "43-post-cutoff",         "post_use_cutoff": True, "post_cutoff": 0.015},

    # ── Phase 5: Query transforms ─────────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "50-transform-rewrite",      "use_query_rewrite": True},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "51-transform-hyde",         "use_hyde": True},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "52-transform-decomposition","use_query_decomposition": True},

    # ── Phase 6: Context processing ───────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "60-context-dedup",          "use_dedup": True, "dedup_threshold": 0.80},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "61-context-consolidation",  "use_llm_consolidation": True},

    # ── Phase 7: Prompt variants ──────────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "70-prompt-strict",      "prompt_pretext": PROMPT_STRICT},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "71-prompt-soft",        "prompt_pretext": PROMPT_SOFT},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "72-prompt-permissive",  "prompt_pretext": PROMPT_PERMISSIVE},

    # ── Phase 8: Response mode ────────────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown", "config_name": "80-mode-refine",        "response_mode": "REFINE"},
    {**BASE, "ind_name": "sentences-markdown", "config_name": "81-mode-tree-summarize","response_mode": "TREE_SUMMARIZE"},

    # ── Phase 9: Combined ─────────────────────────────────────────────────────
    {**BASE, "ind_name": "sentences-markdown",       "config_name": "90-combined-best-guess",
     "retriever_with_keywords": True, "retriever_query_variants": 2, "retriever_top_k": 10,
     "post_use_rerank": True, "post_rerank_top_n": 5, "post_use_reorder": True,
     "use_query_rewrite": True, "use_dedup": True, "dedup_threshold": 0.80,
     "prompt_pretext": PROMPT_SOFT},

    {**BASE, "ind_name": "sentence_window-markdown", "config_name": "91-combined-sentwindow-best",
     "retriever_with_keywords": True, "retriever_query_variants": 2, "retriever_top_k": 10,
     "post_use_rerank": True, "post_rerank_top_n": 5, "post_use_reorder": True,
     "use_query_rewrite": True, "use_dedup": True, "dedup_threshold": 0.80,
     "prompt_pretext": PROMPT_SOFT},

    {**BASE, "ind_name": "sentences-markdown",       "config_name": "92-combined-best-permissive",
     "retriever_with_keywords": True, "retriever_query_variants": 2, "retriever_top_k": 10,
     "post_use_rerank": True, "post_rerank_top_n": 5, "post_use_reorder": True,
     "use_query_rewrite": True, "use_dedup": True, "dedup_threshold": 0.80,
     "prompt_pretext": PROMPT_PERMISSIVE},

    {**BASE, "ind_name": "sentences-markdown",       "config_name": "93-kitchen-sink",
     "retriever_with_keywords": True, "retriever_query_variants": 3, "retriever_top_k": 15,
     "post_use_rerank": True, "post_rerank_top_n": 5, "post_use_reorder": True,
     "use_query_rewrite": True, "use_dedup": True, "dedup_threshold": 0.75,
     "use_llm_consolidation": True, "prompt_pretext": PROMPT_SOFT, "response_mode": "REFINE"},
]

GLOBAL_SNAME = "Overnight-30q-v1"

def full_configs_test(configs, questions, truths):
    start = time.time()
    error_configs = []
    key = key_from_file(KEYFILE)
    for cfg in configs:
        cfg = cfg.copy()
        is_simple_only = cfg.pop("_simple_only", False)
        if not (cfg.get("config_name", "?") == "43-post-cutoff"):
            print(f"SKIPPED {cfg.get('config_name', '?')}")
            continue
        try:
            print("")
            print("=" * 60)
            logger.info("Starting config: {}", cfg.get("config_name", "?"))
            print("=" * 60)
            rag = RAGPipeline(APIKey=key, **cfg, **EVAL_SETTINGS, session_name=GLOBAL_SNAME)
            if is_simple_only:
                rag.batch_simple_query(questions, truths)
            else:
                rag.batch_RAG_query(questions, references=truths)
            del rag
        except Exception as e:
            logger.error("Config {} failed entirely: {}", cfg.get("config_name", "?"), e)
            error_configs.append(cfg.get("config_name", "?"))
            continue
    logger.error("Following Configs could not be tested: \n{}", error_configs)
    passed_time_min = (time.time() - start)/60
    passed_time_h = (passed_time_min) / 60
    logger.info("Time passed {} min / {} h", passed_time_min, passed_time_h)

def rep_test_bad():
    config = CONFIGS[0]
    question = QUESTIONS[3]
    reference = GROUND_TRUTH[3]
    key = key_from_file(KEYFILE)

    is_simple_only = config.pop("_simple_only", False)

    rag = RAGPipeline(APIKey=key, **config, **EVAL_SETTINGS, session_name="00000-test")
    rag.batch_simple_query([question], [reference])

def rep_test_good():
    config = CONFIGS[23]
    question = QUESTIONS[3]
    reference = GROUND_TRUTH[3]
    key = key_from_file(KEYFILE)

    is_simple_only = config.pop("_simple_only", False)

    rag = RAGPipeline(APIKey=key, **config, **EVAL_SETTINGS, session_name="00000-test")
    rag.batch_RAG_query([question], [reference])

if __name__ == "__main__":
    #full_configs_test(configs=CONFIGS, questions=QUESTIONS, truths=GROUND_TRUTH)
    rep_test_good()