#!/usr/bin/env python3
"""Generate depth files using the ApproximateBackwardChainingGrounder.

For each split (train, valid, test), computes the minimum proof depth of each
fact. A fact's depth is the minimum number of backward chaining steps needed
to prove it from the training facts.

Runs the framework grounder at each depth d=1..max_depth and records when
each query first becomes provable.

Output:
  {split}_depths_keras.txt  — one line per query: "triple depth"
                              depth=-1 means unprovable

Usage:
    cd keras-ns/experiments
    python generate_depths.py -d countries_s3 --max_depth 6
    python generate_depths.py -d countries_s3 --max_depth 4 --splits train test
    python generate_depths.py -d wn18rr --max_depth 2 --n_queries 200
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

keras_ns_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, keras_ns_root)
sys.path.insert(0, os.path.join(keras_ns_root, 'experiments'))

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ── Parsing ──────────────────────────────────────────────────────────────

def parse_functional(path: str) -> List[Tuple[str, str, str]]:
    """Parse functional format: rel(subj,obj)."""
    facts = []
    pat = re.compile(r"(\w+)\((\w+),(\w+)\)\.")
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                facts.append((m.group(1), m.group(2), m.group(3)))
    return facts


def load_rules(path: str, num_rules: int = 9999) -> list:
    """Load rules as framework Rule objects."""
    from ns_lib.logic.commons import Rule
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rule = Rule(s=line)
                rules.append(rule)
            except Exception:
                continue
            if len(rules) >= num_rules:
                break
    return rules


def build_domains(facts: List[Tuple[str, str, str]]) -> dict:
    """Build a single default domain from all constants in facts."""
    from ns_lib.logic.commons import Domain
    all_constants = sorted(set(c for f in facts for c in f[1:]))
    return {"default": Domain("default", all_constants)}


def load_domain_file(path: str) -> dict:
    """Load domain file and return Dict[str, Domain]."""
    from ns_lib.logic.commons import Domain
    domains = {}
    all_constants = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = parts[0]
            constants = sorted(set(parts[1:]))
            domains[name] = Domain(name, constants)
            all_constants.extend(constants)
    # Also add a default domain with ALL constants (for rules with default var2domain)
    domains["default"] = Domain("default", sorted(set(all_constants)))
    return domains


# ── Provable query extraction ──────────────────────────────────────────

def extract_proven_queries(result: dict, query_set: Set[Tuple]) -> Set[Tuple]:
    """Extract queries proven by the grounder from the grounding result.

    After pruning, any grounding whose head atom matches a query means
    that query is provable.
    """
    proven = set()
    for rule_name, rule_groundings in result.items():
        for grounding in rule_groundings.groundings:
            head_tuple = grounding[0]   # tuple of head atoms
            for atom in head_tuple:
                if atom in query_set:
                    proven.add(atom)
    return proven


# ── Output ─────────────────────────────────────────────────────────────

def write_depth_file(queries, depths, path):
    """Write depth file: one line per query 'triple depth'."""
    with open(path, 'w') as f:
        for q, d in zip(queries, depths):
            triple_str = f"{q[0]}({q[1]},{q[2]})"
            f.write(f"{triple_str} {d}\n")


def print_distribution(depths, split_name):
    """Print depth distribution summary."""
    counts = defaultdict(int)
    for d in depths:
        counts[d] += 1

    N = len(depths)
    provable = N - counts.get(-1, 0)

    print(f"\n  {split_name} depth distribution ({provable}/{N} provable, "
          f"{provable/N:.1%}):")

    for d in sorted(counts):
        label = f"d={d}" if d >= 0 else "unprovable"
        pct = counts[d] / N * 100
        bar = '#' * int(pct / 2)
        print(f"    {label:>12}: {counts[d]:>6} ({pct:5.1f}%) {bar}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate depth files using keras-ns backward chaining grounder"
    )
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help="Dataset name (directory under data/)")
    parser.add_argument(
        '--data_path', type=str, default=DATA_PATH,
        help=f"Base data path (default: {DATA_PATH})")
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'valid', 'test'],
        help="Splits to process (default: train valid test)")
    parser.add_argument(
        '--max_depth', type=int, default=6,
        help="Maximum backward chaining depth (default: 6)")
    parser.add_argument(
        '--num_rules', type=int, default=9999,
        help="Max number of rules to load (default: 9999 = all)")
    parser.add_argument(
        '--n_queries', type=int, default=None,
        help="Limit number of queries per split (default: all)")
    parser.add_argument(
        '--train_file', type=str, default='train.txt',
        help="Training file name")
    parser.add_argument(
        '--rules_file', type=str, default='rules.txt',
        help="Rules file name")
    parser.add_argument(
        '--max_unknown', type=int, default=1,
        help="Max unknown body atoms per grounding in intermediate steps "
             "(default: 1)")
    args = parser.parse_args()

    from ns_lib.grounding.backward_chaining_grounder import (
        ApproximateBackwardChainingGrounder)

    split_files = {
        'train': args.train_file,
        'valid': 'valid.txt',
        'test': 'test.txt',
    }

    data_dir = os.path.join(args.data_path, args.dataset)

    print(f"{'='*60}")
    print(f"GENERATE DEPTHS — keras-ns (backward chaining grounder)")
    print(f"{'='*60}")
    print(f"Dataset:      {args.dataset}")
    print(f"Data path:    {args.data_path}")
    print(f"Splits:       {args.splits}")
    print(f"Max depth:    {args.max_depth}")
    print(f"Num rules:    {args.num_rules}")
    print(f"N queries:    {args.n_queries or 'all'}")
    print(f"Max unknown:  {args.max_unknown}")
    print(f"{'='*60}")

    # Load training data
    train_path = os.path.join(data_dir, args.train_file)
    rules_path = os.path.join(data_dir, args.rules_file)

    print(f"\nLoading dataset '{args.dataset}'...")
    train_facts = parse_functional(train_path)

    # Load rules as framework Rule objects
    all_rules = load_rules(rules_path, args.num_rules)
    # Filter to ≤2-body rules
    rules = [r for r in all_rules if len(r.body) <= 2]

    # Build domains
    domain_file = os.path.join(data_dir, 'domain2constants.txt')
    if os.path.exists(domain_file):
        domains = load_domain_file(domain_file)
        print(f"  Loaded domains from domain2constants.txt: "
              f"{', '.join(f'{n}({len(d.constants)})' for n, d in domains.items())}")
    else:
        domains = build_domains(train_facts)

    n_constants = len(domains.get("default",
                                  list(domains.values())[0]).constants)
    print(f"  Constants:  {n_constants}")
    print(f"  Facts:      {len(train_facts)}")
    print(f"  Rules:      {len(rules)}")
    for r in rules:
        body_str = ", ".join(f"{b[0]}({b[1]},{b[2]})" for b in r.body)
        head_str = f"{r.head[0][0]}({r.head[0][1]},{r.head[0][2]})"
        print(f"    {r.name}: {body_str} -> {head_str} "
              f"[c={r.weight:.3f}]")

    # Collect all queries across splits
    all_split_queries = {}
    all_queries_list = []
    for split in args.splits:
        filename = split_files.get(split)
        if filename is None:
            continue
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"\nSplit '{split}': file not found ({filepath}), skipping")
            continue
        split_facts = sorted(parse_functional(filepath))
        if args.n_queries is not None and args.n_queries < len(split_facts):
            split_facts = split_facts[:args.n_queries]
        all_split_queries[split] = split_facts
        all_queries_list.extend(split_facts)

    unique_queries = set(all_queries_list)
    query_depths = {}  # query tuple -> depth
    proven_queries = set()

    # For each depth, run the grounder on remaining queries
    print(f"\nRunning backward chaining grounder at each depth "
          f"(1..{args.max_depth})...")

    total_time = 0
    for d in range(1, args.max_depth + 1):
        remaining = [q for q in all_queries_list if q not in proven_queries]
        if not remaining:
            print(f"\n  Depth {d}: all queries assigned, stopping early")
            break

        t0 = time.perf_counter()

        grounder = ApproximateBackwardChainingGrounder(
            rules=rules,
            facts=train_facts,
            domains=domains,
            max_unknown_fact_count=args.max_unknown,
            max_unknown_fact_count_last_step=0,
            num_steps=d,
            prune_incomplete_proofs=True,
        )

        result = grounder.ground(facts=train_facts, queries=remaining)

        newly_proven = extract_proven_queries(result, set(remaining))
        for q in newly_proven:
            if q not in query_depths:
                query_depths[q] = d
        proven_queries.update(newly_proven)

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        n_remaining = len(all_queries_list) - len(proven_queries)
        print(f"\n  Depth {d}: +{len(newly_proven)} queries proven, "
              f"{n_remaining} remaining, {elapsed:.2f}s")

    print(f"\n  Total grounder time: {total_time:.2f}s")
    n_unique = len(unique_queries)
    print(f"  Proven: {len(proven_queries)}/{n_unique} unique queries "
          f"({len(proven_queries)/max(n_unique,1):.1%})")

    # Write depth files for each split
    for split in args.splits:
        if split not in all_split_queries:
            print(f"\nSplit '{split}': no queries found, skipping")
            continue

        split_facts = all_split_queries[split]

        print(f"\n{'='*60}")
        print(f"Split: {split} ({len(split_facts)} queries)")
        print(f"{'='*60}")

        depths = [query_depths.get(q, -1) for q in split_facts]

        print_distribution(depths, split)

        output_path = os.path.join(data_dir, f'{split}_depths_keras.txt')
        write_depth_file(split_facts, depths, output_path)
        print(f"  Saved to: {output_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
