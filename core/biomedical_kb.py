"""
hleo_v1/core/biomedical_kb.py
==============================
Biomedical Knowledge Base — Phase 1 & 2 of the Semantic Search Engine.

Provides:
  - Multi-language alias dictionaries (EN + IT) for drugs, conditions, symptoms
  - MeSH descriptor mappings
  - Biomedical knowledge graph (drug → mechanism → effect → related drug)
  - lookup_entity()  — hybrid entity recognition (dictionary → fuzzy → LLM)
  - get_neighbors()  — 1-hop or 2-hop graph traversal for expansion
  - get_mesh_terms() — MeSH descriptors for an entity

All structures are pure Python; no external dependencies required.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Normalisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lower-case, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_text.lower().strip())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DRUG ALIASES  (normalized_name → list of aliases, EN + IT)
# ─────────────────────────────────────────────────────────────────────────────

DRUG_ALIASES: dict[str, list[str]] = {
    # ── 5-alpha reductase inhibitors ─────────────────────────────────────────
    "finasteride": [
        "finasteride", "propecia", "proscar", "finpecia", "fincar",
        "5-alpha reductase inhibitor", "5ari",
        # Italian / layman
        "finasteride", "propecia",
    ],
    "dutasteride": [
        "dutasteride", "avodart", "avidart", "duodart",
        "dual 5-alpha reductase inhibitor", "dual 5ari",
        # Italian
        "dutasteride", "avodart",
    ],
    # ── Vasodilators / hair growth ────────────────────────────────────────────
    "minoxidil": [
        "minoxidil", "rogaine", "regaine", "loniten", "mintop",
        "topical minoxidil", "oral minoxidil",
        # Italian
        "minoxidil", "rogaine",
    ],
    # ── Retinoids ─────────────────────────────────────────────────────────────
    "isotretinoin": [
        "isotretinoin", "accutane", "roaccutane", "claravis", "amnesteem",
        "13-cis-retinoic acid", "isotrex", "retinoic acid",
        # Italian
        "isotretinoina", "roaccutane", "accutane",
    ],
    # ── Corticosteroids ───────────────────────────────────────────────────────
    "prednisone": [
        "prednisone", "prednisolone", "cortisone", "corticosterone",
        "methylprednisolone", "dexamethasone", "hydrocortisone",
        "betamethasone", "triamcinolone", "budesonide", "fluticasone",
        "systemic corticosteroid", "glucocorticoid", "steroid",
        # Italian
        "cortisone", "cortisonico", "steroide", "prednisone", "prednisolone",
        "desametasone", "metilprednisolone",
    ],
    # ── Anti-androgens ────────────────────────────────────────────────────────
    "spironolactone": [
        "spironolactone", "aldactone", "spiractin",
        "anti-androgen", "androgen blocker",
        # Italian
        "spironolattone", "aldactone",
    ],
    "bicalutamide": [
        "bicalutamide", "casodex", "cosudex",
        "non-steroidal anti-androgen",
        # Italian
        "bicalutamide", "casodex",
    ],
    # ── NSAIDs ────────────────────────────────────────────────────────────────
    "ibuprofen": [
        "ibuprofen", "nurofen", "advil", "motrin", "brufen",
        "nsaid", "anti-inflammatory", "cox inhibitor",
        # Italian
        "ibuprofene", "brufen", "nurofen", "antinfiammatorio",
    ],
    "naproxen": [
        "naproxen", "aleve", "naprosyn",
        # Italian
        "naprossene",
    ],
    "diclofenac": [
        "diclofenac", "voltaren", "voltarol",
        # Italian
        "diclofenac", "voltaren",
    ],
    # ── Immunosuppressants / DMARDs ───────────────────────────────────────────
    "methotrexate": [
        "methotrexate", "mtx", "rheumatrex", "trexall",
        "disease-modifying antirheumatic drug", "dmard",
        # Italian
        "metotrexato", "methotrexate",
    ],
    # ── Antifungals ───────────────────────────────────────────────────────────
    "ketoconazole": [
        "ketoconazole", "nizoral", "extina",
        "antifungal shampoo",
        # Italian
        "ketoconazolo", "nizoral",
    ],
    # ── Statins ───────────────────────────────────────────────────────────────
    "atorvastatin": [
        "atorvastatin", "lipitor", "torvast", "statin",
        "hmg-coa reductase inhibitor",
        # Italian
        "atorvastatina", "lipitor",
    ],
    # ── Anticoagulants ────────────────────────────────────────────────────────
    "warfarin": [
        "warfarin", "coumadin", "jantoven",
        "anticoagulant", "blood thinner",
        # Italian
        "warfarin", "coumadin", "anticoagulante",
    ],
    # ── Hormones ─────────────────────────────────────────────────────────────
    "testosterone": [
        "testosterone", "androgel", "testim", "androderm",
        "testosterone replacement therapy", "trt",
        # Italian
        "testosterone", "terapia ormonale sostitutiva",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONDITION ALIASES
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_ALIASES: dict[str, list[str]] = {
    "androgenetic alopecia": [
        "androgenetic alopecia", "aga", "male pattern baldness", "female pattern baldness",
        "androgenic alopecia", "pattern hair loss", "mpb", "fpb",
        "alopecia androgenetica",
        # Italian
        "calvizie", "alopecia androgenetica", "caduta dei capelli", "diradamento",
    ],
    "telogen effluvium": [
        "telogen effluvium", "te", "hair shedding", "diffuse hair loss",
        "acute telogen effluvium", "chronic telogen effluvium",
        # Italian
        "effluvio telogenico", "caduta diffusa", "caduta massiccia capelli",
    ],
    "alopecia areata": [
        "alopecia areata", "aa", "patchy hair loss", "spot baldness",
        "alopecia totalis", "alopecia universalis",
        # Italian
        "alopecia areata", "alopecia a chiazze",
    ],
    "post-finasteride syndrome": [
        "post-finasteride syndrome", "pfs", "post finasteride syndrome",
        "finasteride side effects persistent", "persistent sexual side effects",
        # Italian
        "sindrome post-finasteride", "effetti collaterali persistenti finasteride",
    ],
    "benign prostatic hyperplasia": [
        "benign prostatic hyperplasia", "bph", "benign prostatic hypertrophy",
        "enlarged prostate", "prostate enlargement",
        # Italian
        "ipertrofia prostatica benigna", "ipb", "prostata ingrossata",
    ],
    "acne vulgaris": [
        "acne vulgaris", "acne", "nodulocystic acne", "severe acne",
        "cystic acne", "comedones", "papules",
        # Italian
        "acne", "acne vulgaris", "acne cistica", "brufoli",
    ],
    "osteoporosis": [
        "osteoporosis", "bone density loss", "bone loss", "low bone density",
        "osteopenia", "fragility fracture",
        # Italian
        "osteoporosi", "perdita di massa ossea", "osteopenia",
    ],
    "erectile dysfunction": [
        "erectile dysfunction", "ed", "impotence", "sexual dysfunction",
        "male sexual dysfunction",
        # Italian
        "disfunzione erettile", "impotenza", "problemi erettili",
    ],
    "sexual dysfunction": [
        "sexual dysfunction", "loss of libido", "decreased libido",
        "anorgasmia", "ejaculatory dysfunction", "decreased sexual desire",
        # Italian
        "disfunzione sessuale", "calo della libido", "problemi sessuali",
        "mancanza di desiderio", "eiaculazione",
    ],
    "arthritis": [
        "arthritis", "rheumatoid arthritis", "ra", "osteoarthritis", "oa",
        "joint inflammation", "polyarthritis",
        # Italian
        "artrite", "artrite reumatoide", "artrosi", "infiammazione articolare",
    ],
    "depression": [
        "depression", "major depressive disorder", "mdd", "depressive disorder",
        "mood disorder",
        # Italian
        "depressione", "disturbo depressivo",
    ],
    "anxiety": [
        "anxiety", "anxiety disorder", "generalized anxiety", "gad",
        # Italian
        "ansia", "disturbo ansioso",
    ],
    "cognitive impairment": [
        "cognitive impairment", "memory loss", "cognitive decline", "brain fog",
        "concentration problems", "memory problems",
        # Italian
        "perdita di memoria", "nebbia cognitiva", "problemi di memoria",
        "difficolta di concentrazione", "calo cognitivo",
    ],
    "hypothyroidism": [
        "hypothyroidism", "underactive thyroid", "thyroid deficiency",
        # Italian
        "ipotiroidismo", "tiroide",
    ],
    "cardiovascular disease": [
        "cardiovascular disease", "heart disease", "cardiac disease",
        "coronary artery disease", "atherosclerosis",
        # Italian
        "malattia cardiovascolare", "cardiopatia", "problema cardiaco",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SYMPTOM / ADVERSE EFFECT ALIASES
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_ALIASES: dict[str, list[str]] = {
    "hair loss": [
        "hair loss", "hair shedding", "alopecia", "hair fall", "hair thinning",
        "diffuse shedding", "initial shedding",
        # Italian
        "caduta capelli", "perdita capelli", "diradamento capelli", "caduta",
        "perdita di capelli",
    ],
    "sexual dysfunction": [
        "sexual dysfunction", "libido loss", "low libido", "decreased libido",
        "erectile dysfunction", "impotence", "ejaculatory problems",
        "sexual side effects",
        # Italian
        "problemi sessuali", "calo libido", "disfunzione sessuale",
        "perdita libido", "problemi erettili",
    ],
    "depression": [
        "depression", "depressive symptoms", "low mood", "dysphoria",
        # Italian
        "depressione", "umore basso", "tristezza persistente",
    ],
    "memory loss": [
        "memory loss", "memory problems", "forgetfulness", "brain fog",
        "cognitive impairment", "difficulty concentrating",
        # Italian
        "perdita di memoria", "problemi di memoria", "nebbia mentale",
        "difficolta concentrazione", "vuoti di memoria",
    ],
    "joint pain": [
        "joint pain", "arthralgia", "myalgia", "musculoskeletal pain",
        "muscle pain", "joint ache",
        # Italian
        "dolori articolari", "dolori muscolari", "dolore alle articolazioni",
        "artralgia",
    ],
    "fatigue": [
        "fatigue", "tiredness", "exhaustion", "chronic fatigue",
        # Italian
        "stanchezza", "affaticamento", "spossatezza",
    ],
    "gynecomastia": [
        "gynecomastia", "gynaecomastia", "breast enlargement", "breast tenderness",
        # Italian
        "ginecomastia", "ingrossamento seno maschile",
    ],
    "dry skin": [
        "dry skin", "xerosis", "skin dryness", "cheilitis", "dry lips",
        # Italian
        "pelle secca", "secchezza cutanea", "labbra secche",
    ],
    "cardiac effects": [
        "cardiac effects", "heart palpitations", "tachycardia", "fluid retention",
        "water retention", "edema",
        # Italian
        "effetti cardiaci", "palpitazioni", "tachicardia", "ritenzione idrica",
        "edema", "effetti sul cuore", "problemi cardiaci",
    ],
    "nausea": [
        "nausea", "vomiting", "gastrointestinal side effects",
        # Italian
        "nausea", "vomito", "mal di stomaco",
    ],
    "bone loss": [
        "bone loss", "osteoporosis", "fracture risk", "bone density reduction",
        # Italian
        "perdita ossea", "osteoporosi", "riduzione densita ossea",
    ],
    "photosensitivity": [
        "photosensitivity", "sun sensitivity", "sunburn",
        # Italian
        "fotosensibilita", "sensibilita al sole",
    ],
    "teratogenicity": [
        "teratogenicity", "birth defects", "fetal abnormalities", "pregnancy risk",
        # Italian
        "teratogenicita", "difetti congeniti", "rischio in gravidanza",
    ],
    "knee pain": [
        "knee pain", "knee replacement pain", "post-arthroplasty pain",
        "prosthesis pain", "knee arthroplasty",
        # Italian
        "dolore ginocchio", "protesi ginocchio", "dolore dopo protesi",
        "artroprotesi", "protesi al ginocchio",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MeSH DESCRIPTOR MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

MESH_MAP: dict[str, list[str]] = {
    "finasteride":              ["Finasteride", "5-alpha Reductase Inhibitors"],
    "dutasteride":              ["Dutasteride", "5-alpha Reductase Inhibitors"],
    "minoxidil":                ["Minoxidil", "Vasodilator Agents"],
    "isotretinoin":             ["Isotretinoin", "Retinoids", "Keratolytic Agents"],
    "prednisone":               ["Prednisone", "Glucocorticoids", "Adrenal Cortex Hormones"],
    "spironolactone":           ["Spironolactone", "Mineralocorticoid Receptor Antagonists"],
    "ibuprofen":                ["Ibuprofen", "Anti-Inflammatory Agents, Non-Steroidal",
                                 "Cyclooxygenase Inhibitors"],
    "methotrexate":             ["Methotrexate", "Antimetabolites, Antineoplastic",
                                 "Antirheumatic Agents"],
    "androgenetic alopecia":    ["Alopecia", "Alopecia, Androgenetic"],
    "telogen effluvium":        ["Alopecia", "Hair Diseases"],
    "alopecia areata":          ["Alopecia Areata"],
    "post-finasteride syndrome":["Post-Finasteride Syndrome", "Sexual Dysfunction, Physiological"],
    "benign prostatic hyperplasia": ["Prostatic Hyperplasia"],
    "acne vulgaris":            ["Acne Vulgaris"],
    "osteoporosis":             ["Osteoporosis", "Bone Density Conservation Agents"],
    "erectile dysfunction":     ["Erectile Dysfunction"],
    "sexual dysfunction":       ["Sexual Dysfunction, Physiological", "Libido"],
    "arthritis":                ["Arthritis", "Arthritis, Rheumatoid", "Osteoarthritis"],
    "depression":               ["Depression", "Depressive Disorder"],
    "cognitive impairment":     ["Cognitive Dysfunction", "Memory Disorders"],
    "hair loss":                ["Alopecia", "Hair Diseases", "Hair Follicle"],
    "joint pain":               ["Arthralgia", "Myalgia"],
    "cardiac effects":          ["Tachycardia", "Edema", "Fluid Retention"],
    "bone loss":                ["Bone Density", "Osteoporosis"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  BIOMEDICAL KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
# Structure:  entity → { relation_type → [target_entities] }
# Relations:  is_a, mechanism, inhibits, treats, adverse_effects,
#             related_drug, associated_with, causes

KNOWLEDGE_GRAPH: dict[str, dict[str, list[str]]] = {
    "finasteride": {
        "is_a":            ["5-alpha reductase inhibitor", "5ari"],
        "inhibits":        ["5-alpha reductase type II", "dht"],
        "treats":          ["androgenetic alopecia", "benign prostatic hyperplasia"],
        "adverse_effects": ["sexual dysfunction", "erectile dysfunction",
                            "gynecomastia", "depression", "hair loss",
                            "post-finasteride syndrome"],
        "related_drug":    ["dutasteride", "minoxidil", "spironolactone"],
        "mechanism":       ["dihydrotestosterone reduction", "androgen suppression"],
    },
    "dutasteride": {
        "is_a":            ["dual 5-alpha reductase inhibitor", "5ari"],
        "inhibits":        ["5-alpha reductase type I", "5-alpha reductase type II", "dht"],
        "treats":          ["androgenetic alopecia", "benign prostatic hyperplasia"],
        "adverse_effects": ["hair loss", "telogen effluvium", "sexual dysfunction",
                            "gynecomastia", "depression"],
        "related_drug":    ["finasteride", "minoxidil"],
        "mechanism":       ["dual DHT inhibition", "androgen suppression"],
    },
    "minoxidil": {
        "is_a":            ["potassium channel opener", "vasodilator"],
        "treats":          ["androgenetic alopecia", "telogen effluvium", "hair loss"],
        "adverse_effects": ["cardiac effects", "hair loss", "facial hypertrichosis",
                            "fluid retention", "tachycardia"],
        "related_drug":    ["finasteride", "dutasteride", "spironolactone"],
        "mechanism":       ["vasodilation", "hair follicle stimulation", "anagen prolongation"],
    },
    "isotretinoin": {
        "is_a":            ["retinoid", "vitamin A derivative"],
        "treats":          ["acne vulgaris", "nodulocystic acne"],
        "adverse_effects": ["dry skin", "teratogenicity", "depression", "memory loss",
                            "hair loss", "joint pain", "photosensitivity",
                            "elevated triglycerides", "liver toxicity"],
        "related_drug":    ["tretinoin", "adapalene", "tazarotene"],
        "mechanism":       ["sebum reduction", "sebaceous gland involution",
                            "retinoid receptor agonism"],
    },
    "prednisone": {
        "is_a":            ["glucocorticoid", "corticosteroid"],
        "treats":          ["inflammation", "arthritis", "autoimmune disease",
                            "asthma", "alopecia areata"],
        "adverse_effects": ["osteoporosis", "bone loss", "weight gain",
                            "hypertension", "diabetes", "immunosuppression",
                            "cushing syndrome", "adrenal suppression"],
        "related_drug":    ["methylprednisolone", "dexamethasone", "hydrocortisone"],
        "mechanism":       ["glucocorticoid receptor activation", "anti-inflammation"],
    },
    "spironolactone": {
        "is_a":            ["anti-androgen", "mineralocorticoid antagonist"],
        "treats":          ["androgenetic alopecia", "female pattern hair loss", "acne"],
        "adverse_effects": ["hyperkalemia", "gynecomastia", "menstrual irregularities"],
        "related_drug":    ["finasteride", "dutasteride"],
        "mechanism":       ["androgen receptor blockade", "DHT competition"],
    },
    "androgenetic alopecia": {
        "associated_with": ["dihydrotestosterone", "dht", "5-alpha reductase",
                            "androgenic hormone", "testosterone"],
        "treated_by":      ["finasteride", "dutasteride", "minoxidil", "spironolactone",
                            "ketoconazole", "hair transplant"],
        "related_condition":["telogen effluvium", "alopecia areata"],
    },
    "telogen effluvium": {
        "causes":          ["finasteride", "dutasteride", "stress", "nutritional deficiency",
                            "thyroid disorder", "postpartum"],
        "associated_with": ["hair loss", "androgenetic alopecia"],
        "treated_by":      ["minoxidil", "nutritional supplementation"],
    },
    "sexual dysfunction": {
        "causes":          ["finasteride", "dutasteride", "antidepressants", "beta-blockers"],
        "associated_with": ["erectile dysfunction", "post-finasteride syndrome",
                            "libido loss"],
        "related_condition":["depression", "anxiety", "hypogonadism"],
    },
    "post-finasteride syndrome": {
        "causes":          ["finasteride", "dutasteride"],
        "associated_with": ["sexual dysfunction", "depression", "cognitive impairment",
                            "persistent side effects"],
    },
    "osteoporosis": {
        "causes":          ["prednisone", "corticosteroids", "calcium deficiency",
                            "vitamin D deficiency", "hypogonadism"],
        "associated_with": ["bone density loss", "fracture risk"],
        "treated_by":      ["bisphosphonates", "calcium", "vitamin D", "denosumab"],
    },
    "cognitive impairment": {
        "causes":          ["isotretinoin", "statins", "benzodiazepines", "opioids"],
        "associated_with": ["memory loss", "brain fog", "concentration problems"],
    },
    "dht": {
        "is_a":            ["dihydrotestosterone", "androgen"],
        "causes":          ["androgenetic alopecia", "benign prostatic hyperplasia"],
        "inhibited_by":    ["finasteride", "dutasteride"],
    },
    "5-alpha reductase": {
        "converts":        ["testosterone", "dht"],
        "inhibited_by":    ["finasteride", "dutasteride"],
    },
    "hair loss": {
        "types":           ["androgenetic alopecia", "telogen effluvium", "alopecia areata"],
        "causes":          ["finasteride", "dutasteride", "isotretinoin", "stress",
                            "thyroid disorder", "nutritional deficiency"],
        "treated_by":      ["minoxidil", "finasteride", "dutasteride", "spironolactone"],
    },
    "knee replacement": {
        "also_known_as":   ["knee arthroplasty", "total knee replacement", "tkr",
                            "knee prosthesis"],
        "associated_with": ["joint pain", "post-operative pain", "rehabilitation"],
        "complications":   ["chronic pain", "joint stiffness", "prosthesis failure"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Italian → English quick translation for common medical terms
# ─────────────────────────────────────────────────────────────────────────────

_IT_EN_QUICK: dict[str, str] = {
    "caduta":         "hair loss",
    "caduta capelli": "hair loss",
    "perdita":        "loss",
    "perdita capelli":"hair loss",
    "perdita memoria":"memory loss",
    "memoria":        "memory",
    "sessuali":       "sexual",
    "sessuale":       "sexual",
    "problemi":       "problems",
    "dolore":         "pain",
    "dolori":         "pain",
    "protesi":        "prosthesis",
    "ginocchio":      "knee",
    "cortisone":      "corticosteroid",
    "ossa":           "bone",
    "osso":           "bone",
    "articolari":     "joint",
    "articolare":     "joint",
    "cuore":          "cardiac",
    "cardiaco":       "cardiac",
    "stanchezza":     "fatigue",
    "infiammazione":  "inflammation",
    "indotta":        "induced",
    "indotto":        "induced",
    "dopo":           "after",
    "effetti":        "effects",
    "collaterali":    "side effects",
}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Unified lookup index  (alias_norm → (entity_type, canonical_name))
# ─────────────────────────────────────────────────────────────────────────────

# Build at import time so lookups are O(1)
_ALIAS_INDEX: dict[str, tuple[str, str]] = {}

def _build_index() -> None:
    for canonical, aliases in DRUG_ALIASES.items():
        for alias in aliases:
            _ALIAS_INDEX[_norm(alias)] = ("drug", canonical)

    for canonical, aliases in CONDITION_ALIASES.items():
        for alias in aliases:
            _ALIAS_INDEX[_norm(alias)] = ("condition", canonical)

    for canonical, aliases in SYMPTOM_ALIASES.items():
        for alias in aliases:
            _ALIAS_INDEX[_norm(alias)] = ("symptom", canonical)

_build_index()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Public API
# ─────────────────────────────────────────────────────────────────────────────

def lookup_entity(
    text: str,
    fuzzy_threshold: float = 0.82,
) -> list[tuple[str, str, float]]:
    """
    Find all biomedical entities in *text*.

    Strategy (hybrid, cheapest first):
      1. Exact match in alias index (confidence 1.0)
      2. Substring match on individual tokens (confidence 0.95)
      3. Fuzzy match with SequenceMatcher ratio ≥ fuzzy_threshold (lower conf)

    Returns
    -------
    list of (entity_type, canonical_name, confidence)
    e.g. [("drug", "dutasteride", 1.0), ("symptom", "hair loss", 0.95)]
    """
    text_norm = _norm(text)
    found: dict[str, tuple[str, str, float]] = {}  # canonical → best hit

    def _record(etype: str, canonical: str, conf: float) -> None:
        existing = found.get(canonical)
        if existing is None or conf > existing[2]:
            found[canonical] = (etype, canonical, conf)

    # ── 1. Exact match on the full normalised text ────────────────────────────
    if text_norm in _ALIAS_INDEX:
        etype, canonical = _ALIAS_INDEX[text_norm]
        _record(etype, canonical, 1.0)

    # ── 2. Exact match on n-grams (1 to 4 tokens) ────────────────────────────
    tokens = text_norm.split()
    for n in range(1, min(5, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            if gram in _ALIAS_INDEX:
                etype, canonical = _ALIAS_INDEX[gram]
                _record(etype, canonical, 0.95)

    # ── 3. Fuzzy match against all known aliases ──────────────────────────────
    #       (only run if nothing found yet, to keep it cheap)
    if not found:
        for alias_norm, (etype, canonical) in _ALIAS_INDEX.items():
            ratio = _similarity(text_norm, alias_norm)
            if ratio >= fuzzy_threshold:
                _record(etype, canonical, round(ratio * 0.9, 3))  # slightly deflate

    return list(found.values())


def get_neighbors(entity: str, depth: int = 1) -> set[str]:
    """
    Return all entities reachable from *entity* in the knowledge graph
    within *depth* hops (BFS).

    Parameters
    ----------
    entity : canonical entity name (case-insensitive)
    depth  : 1 = direct neighbors; 2 = includes neighbors-of-neighbors

    Returns
    -------
    set of entity names (strings)
    """
    entity_key = entity.lower().strip()
    visited: set[str] = set()
    frontier = {entity_key}

    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            node_data = KNOWLEDGE_GRAPH.get(node)
            if not node_data:
                continue
            for targets in node_data.values():
                for t in targets:
                    t_low = t.lower().strip()
                    if t_low not in visited:
                        next_frontier.add(t_low)
        visited.update(frontier)
        frontier = next_frontier - visited

    return visited | frontier


def get_mesh_terms(entity: str) -> list[str]:
    """
    Return MeSH descriptor strings for *entity*.
    Falls back to checking synonyms if the exact canonical name is missing.
    """
    key = entity.lower().strip()
    # Direct lookup
    terms = MESH_MAP.get(key)
    if terms:
        return list(terms)

    # Try finding via alias index
    hits = lookup_entity(entity)
    for _, canonical, conf in hits:
        if conf >= 0.9 and canonical in MESH_MAP:
            return list(MESH_MAP[canonical])

    return []


def quick_translate_it(text: str) -> str:
    """
    Apply quick Italian→English substitutions on individual tokens.
    Reduces LLM dependency for very common Italian medical terms.
    """
    tokens = text.lower().split()
    result = []
    i = 0
    while i < len(tokens):
        # Try 2-gram first
        if i + 1 < len(tokens):
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in _IT_EN_QUICK:
                result.append(_IT_EN_QUICK[bigram])
                i += 2
                continue
        # Single token
        result.append(_IT_EN_QUICK.get(tokens[i], tokens[i]))
        i += 1
    return " ".join(result)
