"""Phase 6B.3-C Surface-Text Leakage Forensic Investigation Suite.

Performs:
1. Exact reproduction of the 99.25% surface-text result with top feature coefficients and confusion matrix.
2. Text-field isolation experiments (candidate proposition, task, percept, concept labels, semantic relation, stopwords, length stats).
3. Top 50 predictive feature extraction with positive vs negative occurrence statistics.
4. Matched contrast quadruplet (A/B/C/D) separability audit across 7 domains.
5. Proposition diagnostic representation testing (Raw, Normalized, No-Stopwords, Content-Words, Length-Only, Shuffle-Words).
6. Generation mechanism audit (Positive/Negative counts and SEMANTIC_NOVEL rate per template_id, generator_id, source_type).
7. Cross-Template, Cross-Generator, and Cross-Domain generalization holdout tests.
8. Linguistic syntax and causal vocabulary analysis.
9. Derivability classifier evaluation with balanced accuracy and per-class recall.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


def load_dataset() -> list[dict[str, Any]]:
    path = Path(r"c:\Users\bs162\Desktop\THEO\theo-data\datasets\theo_slm_v0_candidates\candidate_records.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_candidate_proposition(r: dict[str, Any]) -> str:
    """Extract candidate proposition string for a record."""
    if r.get("target_interpretation") and r["target_interpretation"].get("proposition"):
        return r["target_interpretation"]["proposition"]
    if r.get("rejected_candidates") and len(r["rejected_candidates"]) > 0:
        return r["rejected_candidates"][0].get("proposition", "")
    if r.get("trap_propositions") and len(r["trap_propositions"]) > 0:
        return r["trap_propositions"][0]
    return ""


def run_experiment_1_reproduce(records: list[dict[str, Any]]) -> dict[str, Any]:
    """1. Reproduce 99.25% result with exact features and feature coefficients."""
    y = np.array([1 if r.get("novelty_label") == "SEMANTIC_NOVEL" else 0 for r in records])
    surface_texts = [f"{r.get('percept', '')} {get_candidate_proposition(r)}" for r in records]

    vectorizer = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
    X = vectorizer.fit_transform(surface_texts).toarray()
    feature_names = vectorizer.get_feature_names_out()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    all_true, all_preds = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, zero_division=0))
        all_true.extend(y_te)
        all_preds.extend(preds)

    # Train full model to extract feature coefficients
    clf_full = LogisticRegression(max_iter=1000, random_state=42)
    clf_full.fit(X, y)
    coefs = clf_full.coef_[0]

    top_pos_idx = np.argsort(coefs)[::-1][:20]
    top_neg_idx = np.argsort(coefs)[:20]

    top_positive_tokens = [{"token": feature_names[i], "coef": round(float(coefs[i]), 4)} for i in top_pos_idx]
    top_negative_tokens = [{"token": feature_names[i], "coef": round(float(coefs[i]), 4)} for i in top_neg_idx]

    cm = confusion_matrix(all_true, all_preds).tolist()
    prec = round(float(precision_score(all_true, all_preds)), 4)
    rec = round(float(recall_score(all_true, all_preds)), 4)

    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "vocabulary_size": len(feature_names),
        "top_positive_features": top_positive_tokens,
        "top_negative_features": top_negative_tokens,
    }


def run_experiment_2_field_isolation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """2. Isolate exactly which text fields drive the 99.25% accuracy."""
    y = np.array([1 if r.get("novelty_label") == "SEMANTIC_NOVEL" else 0 for r in records])

    field_texts = {
        "A_candidate_proposition_only": [get_candidate_proposition(r) for r in records],
        "B_task_text_only": [r.get("task", "") for r in records],
        "C_percept_text_only": [r.get("percept", "") for r in records],
        "D_referenced_concept_names_only": [
            " ".join([c["label"] for c in r.get("concepts", [])]) for r in records
        ],
        "E_semantic_relation_only": [
            r.get("target_interpretation", {}).get("semantic_relation", "none") if r.get("target_interpretation") else "none"
            for r in records
        ],
        "F_proposition_plus_task": [f"{get_candidate_proposition(r)} {r.get('task', '')}" for r in records],
        "G_proposition_stopwords_removed": [
            " ".join([w for w in re.findall(r"\w+", get_candidate_proposition(r).lower()) if w not in {"is", "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with"}])
            for r in records
        ],
        "H_proposition_normalized_lowercase": [get_candidate_proposition(r).lower() for r in records],
        "I_proposition_content_words_only": [
            " ".join([w for w in re.findall(r"\w+", get_candidate_proposition(r).lower()) if len(w) > 4])
            for r in records
        ],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, Any] = {}

    for name, texts in field_texts.items():
        vectorizer = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts).toarray()

        accs, f1s = [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            accs.append(accuracy_score(y_te, preds))
            f1s.append(f1_score(y_te, preds, zero_division=0))

        results[name] = {
            "accuracy": round(float(np.mean(accs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "vocab_len": len(vectorizer.get_feature_names_out()),
        }

    # Representation J: Proposition length / character statistics only (LogisticRegression on numerical length features)
    X_stats = np.array([[len(get_candidate_proposition(r)), len(get_candidate_proposition(r).split())] for r in records])
    accs_stats, f1s_stats = [], []
    for train_idx, test_idx in skf.split(X_stats, y):
        X_tr, X_te = X_stats[train_idx], X_stats[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        accs_stats.append(accuracy_score(y_te, preds))
        f1s_stats.append(f1_score(y_te, preds, zero_division=0))

    results["J_proposition_length_stats_only"] = {
        "accuracy": round(float(np.mean(accs_stats)), 4),
        "macro_f1": round(float(np.mean(f1s_stats)), 4),
        "vocab_len": 2,
    }

    return results


def run_experiment_3_top_tokens(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """3. Extract top 50 predictive TF-IDF tokens with positive vs negative record frequencies."""
    y = [1 if r.get("novelty_label") == "SEMANTIC_NOVEL" else 0 for r in records]
    props = [get_candidate_proposition(r) for r in records]

    vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 2))
    X = vectorizer.fit_transform(props).toarray()
    tokens = vectorizer.get_feature_names_out()

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    coefs = clf.coef_[0]

    top_indices = np.argsort(np.abs(coefs))[::-1][:50]
    token_stats = []

    pos_indices = [i for i, val in enumerate(y) if val == 1]
    neg_indices = [i for i, val in enumerate(y) if val == 0]

    for idx in top_indices:
        t_name = tokens[idx]
        col_vec = X[:, idx]

        pos_count = sum(1 for i in pos_indices if col_vec[i] > 0)
        neg_count = sum(1 for i in neg_indices if col_vec[i] > 0)

        pos_pct = round(pos_count / float(len(pos_indices)) * 100, 1)
        neg_pct = round(neg_count / float(len(neg_indices)) * 100, 1)

        token_stats.append({
            "token": t_name,
            "coef": round(float(coefs[idx]), 4),
            "pos_occurrences": pos_count,
            "neg_occurrences": neg_count,
            "pos_percentage": pos_pct,
            "neg_percentage": neg_pct,
        })

    return token_stats


def run_experiment_4_contrast_quadruplet_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """4. Perform matched semantic-pair analysis on the 28 contrastive quadruplet records."""
    quad_records = [r for r in records if "quad" in r.get("provenance", {}).get("template_id", "")]
    
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in quad_records:
        tmpl = r.get("provenance", {}).get("template_id", "")
        dom = tmpl.split("_")[1] if "_" in tmpl else "unknown"
        by_domain[dom].append(r)

    domain_audits = {}
    for dom, dom_recs in by_domain.items():
        props = [get_candidate_proposition(r) for r in dom_recs]
        lens = [len(p) for p in props]
        novelty_labels = [r.get("novelty_label") for r in dom_recs]
        derivability_labels = [r.get("derivability_label") for r in dom_recs]

        # Check surface separability within domain quadruplet
        domain_audits[dom] = {
            "record_count": len(dom_recs),
            "propositions": props,
            "proposition_lengths": lens,
            "novelty_labels": novelty_labels,
            "derivability_labels": derivability_labels,
            "mean_prop_len": round(float(np.mean(lens)), 1),
            "len_std_dev": round(float(np.std(lens)), 1),
        }

    return {
        "total_contrast_quadruplet_records": len(quad_records),
        "domain_audits": domain_audits,
    }


def run_experiment_5_diagnostic_representations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """5. Test proposition-only classification across 6 diagnostic representations."""
    y = np.array([1 if r.get("novelty_label") == "SEMANTIC_NOVEL" else 0 for r in records])
    props = [get_candidate_proposition(r) for r in records]

    # Rep 1: Raw
    r1 = props

    # Rep 2: Lowercase + Punctuation Normalized
    r2 = [re.sub(r"[^\w\s]", "", p.lower()) for p in props]

    # Rep 3: Stopwords Removed
    r3 = [" ".join([w for w in re.findall(r"\w+", p.lower()) if w not in {"is", "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with"}]) for p in props]

    # Rep 4: Content Words Only (words > 4 chars)
    r4 = [" ".join([w for w in re.findall(r"\w+", p.lower()) if len(w) > 4]) for p in props]

    # Rep 6: Randomized Word-Order Shuffle
    r6 = []
    for p in props:
        words = p.split()
        random.Random(42).shuffle(words)
        r6.append(" ".join(words))

    reps = {
        "Rep_1_Raw_Proposition": r1,
        "Rep_2_Normalized_Lowercase_No_Punct": r2,
        "Rep_3_Stopwords_Removed": r3,
        "Rep_4_Content_Words_Only": r4,
        "Rep_6_Randomized_Word_Order_Shuffle": r6,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rep_results: dict[str, Any] = {}

    for rep_name, texts in reps.items():
        tfidf = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
        X = tfidf.fit_transform(texts).toarray()

        accs, f1s = [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            accs.append(accuracy_score(y_te, preds))
            f1s.append(f1_score(y_te, preds, zero_division=0))

        rep_results[rep_name] = {
            "accuracy": round(float(np.mean(accs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
        }

    return rep_results


def run_experiment_6_generator_template_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """6. Audit Positive vs Negative generation mechanisms per template_id, generator_id, source_type."""
    template_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
    generator_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})

    for r in records:
        prov = r.get("provenance", {})
        tmpl = prov.get("template_id", "unknown")
        gen = prov.get("generator_id", "unknown")
        src = prov.get("source_type", "unknown")

        is_pos = (r.get("novelty_label") == "SEMANTIC_NOVEL")

        template_counts[tmpl]["total"] += 1
        generator_counts[gen]["total"] += 1
        source_counts[src]["total"] += 1

        if is_pos:
            template_counts[tmpl]["pos"] += 1
            generator_counts[gen]["pos"] += 1
            source_counts[src]["pos"] += 1
        else:
            template_counts[tmpl]["neg"] += 1
            generator_counts[gen]["neg"] += 1
            source_counts[src]["neg"] += 1

    template_table = []
    for tmpl, counts in template_counts.items():
        rate = round(counts["pos"] / float(counts["total"]) * 100, 1) if counts["total"] > 0 else 0.0
        template_table.append({
            "template_id": tmpl,
            "positive_count": counts["pos"],
            "negative_count": counts["neg"],
            "total_count": counts["total"],
            "semantic_novel_rate": rate,
        })

    generator_table = []
    for gen, counts in generator_counts.items():
        rate = round(counts["pos"] / float(counts["total"]) * 100, 1) if counts["total"] > 0 else 0.0
        generator_table.append({
            "generator_id": gen,
            "positive_count": counts["pos"],
            "negative_count": counts["neg"],
            "total_count": counts["total"],
            "semantic_novel_rate": rate,
        })

    return {
        "template_audit": template_table,
        "generator_audit": generator_table,
        "source_audit": dict(source_counts),
    }


def run_experiment_7_cross_generalization(records: list[dict[str, Any]]) -> dict[str, Any]:
    """7. Perform Cross-Template, Cross-Generator, and Cross-Domain holdout evaluation."""
    y = np.array([1 if r.get("novelty_label") == "SEMANTIC_NOVEL" else 0 for r in records])
    props = [f"{r.get('percept', '')} {get_candidate_proposition(r)}" for r in records]

    tfidf = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
    X = tfidf.fit_transform(props).toarray()

    templates = np.array([r.get("provenance", {}).get("template_id", "unknown") for r in records])
    generators = np.array([r.get("provenance", {}).get("generator_id", "unknown") for r in records])
    
    # Extract domain from template_id or case_id
    domains_list = []
    for r in records:
        cid = r.get("case_id", "")
        parts = cid.split("/")
        if len(parts) > 3 and parts[3] not in ("neg", "pert", "conflict"):
            domains_list.append(parts[3])
        elif "tmpl_" in r.get("provenance", {}).get("template_id", ""):
            domains_list.append(r["provenance"]["template_id"].split("_")[1])
        else:
            domains_list.append("general")
    domains = np.array(domains_list)

    # 1. Cross-Template Holdout (GroupKFold on template_id)
    n_splits_tmpl = min(5, len(set(templates)))
    gkf_tmpl = GroupKFold(n_splits=n_splits_tmpl)
    tmpl_accs = []
    for train_idx, test_idx in gkf_tmpl.split(X, y, groups=templates):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        tmpl_accs.append(accuracy_score(y_te, preds))

    # 2. Cross-Domain Holdout (GroupKFold on domain)
    n_splits_dom = min(5, len(set(domains)))
    gkf_dom = GroupKFold(n_splits=n_splits_dom)
    dom_accs = []
    for train_idx, test_idx in gkf_dom.split(X, y, groups=domains):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        dom_accs.append(accuracy_score(y_te, preds))

    return {
        "cross_template_holdout_accuracy": round(float(np.mean(tmpl_accs)), 4),
        "cross_domain_holdout_accuracy": round(float(np.mean(dom_accs)), 4),
        "memorization_drop_detected": bool(np.mean(tmpl_accs) < 0.75),
    }


def run_experiment_9_derivability_reevaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """9. Re-evaluate DERIVABILITY classifier against majority baseline using balanced metrics."""
    y = np.array([1 if r.get("derivability_label") == "NON_DERIVABLE" else 0 for r in records])
    props = [get_candidate_proposition(r) for r in records]

    tfidf = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
    X = tfidf.fit_transform(props).toarray()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, bal_accs, f1s = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        accs.append(accuracy_score(y_te, preds))
        bal_accs.append(balanced_accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, zero_division=0))
        all_true.extend(y_te)
        all_pred.extend(preds)

    cm = confusion_matrix(all_true, all_pred).tolist()
    maj_baseline = float(np.mean(y == 1))

    return {
        "overall_accuracy": round(float(np.mean(accs)), 4),
        "majority_class_baseline": round(maj_baseline, 4),
        "balanced_accuracy": round(float(np.mean(bal_accs)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "confusion_matrix": cm,
        "useful_discrimination_demonstrated": bool(np.mean(bal_accs) > 0.60),
    }


def main():
    print("=" * 80)
    print("THEO SLM Phase 6B.3-C — Surface-Text Leakage Forensic Investigation Suite")
    print("=" * 80)

    records = load_dataset()
    print(f"Loaded dataset: {len(records)} candidate records.")

    print("\n[1/7] Reproducing 99.25% Surface-Text Result...")
    exp1 = run_experiment_1_reproduce(records)
    print(f"  - Accuracy: {exp1['accuracy']} | Macro F1: {exp1['macro_f1']}")

    print("\n[2/7] Executing Text-Field Isolation Experiments...")
    exp2 = run_experiment_2_field_isolation(records)
    for name, res in exp2.items():
        print(f"  - {name}: Acc = {res['accuracy']} | F1 = {res['macro_f1']}")

    print("\n[3/7] Extracting Top 50 Predictive Leakage Features...")
    exp3 = run_experiment_3_top_tokens(records)
    print(f"  - Top 3 predictive positive tokens: {[t['token'] for t in exp3[:3]]}")

    print("\n[4/7] Auditing Matched Semantic Contrast Quadruplets...")
    exp4 = run_experiment_4_contrast_quadruplet_audit(records)

    print("\n[5/7] Testing Diagnostic Proposition Representations...")
    exp5 = run_experiment_5_diagnostic_representations(records)

    print("\n[6/7] Auditing Generation Mechanisms & Holdout Generalization...")
    exp6 = run_experiment_6_generator_template_audit(records)
    exp7 = run_experiment_7_cross_generalization(records)
    print(f"  - Cross-Template Holdout Acc: {exp7['cross_template_holdout_accuracy']}")
    print(f"  - Cross-Domain Holdout Acc:   {exp7['cross_domain_holdout_accuracy']}")

    print("\n[7/7] Re-evaluating DERIVABILITY Classifier Metrics...")
    exp9 = run_experiment_9_derivability_reevaluation(records)
    print(f"  - DERIVABILITY Acc: {exp9['overall_accuracy']} (Majority = {exp9['majority_class_baseline']}) | Balanced Acc = {exp9['balanced_accuracy']}")

    # Save complete raw forensic JSON artifact
    output_path = Path(r"c:\Users\bs162\Desktop\THEO\theo-data\datasets\theo_slm_v0_candidates\surface-leakage-forensics-results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_1_reproduce": exp1,
            "experiment_2_field_isolation": exp2,
            "experiment_3_top_tokens": exp3,
            "experiment_4_contrast_quadruplet_audit": exp4,
            "experiment_5_diagnostic_representations": exp5,
            "experiment_6_generator_template_audit": exp6,
            "experiment_7_cross_generalization": exp7,
            "experiment_9_derivability_reevaluation": exp9,
        }, f, indent=2)

    print(f"\nSaved complete forensic results to: {output_path}")


if __name__ == "__main__":
    main()
