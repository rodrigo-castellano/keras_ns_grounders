import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

from kge_inference import KGEInference  # noqa: E402


def infer_default_signature(dataset_name: str) -> str:
    """Return the default run signature for a dataset."""
    if dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
        return f"{dataset_name}-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt"
    if dataset_name == "family":
        return "kinship_family-backward_0_1-no_reasoner-rotate-True-256-256-4-rules.txt"
    if dataset_name == "wn18rr":
        return "wn18rr-backward_0_1-no_reasoner-rotate-True-256-256-1-rules.txt"
    # Fallback to a generic signature
    return f"{dataset_name}-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt"


def ensure_parent_dir(filepath: str) -> None:
    """Create parent directory for a file if it does not exist."""
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def rank_top_k(entries: Iterable[Tuple[Tuple[str, str, str], float]], k: int) -> List[Tuple[Tuple[str, str, str], float]]:
    """Return the top-k entries by score."""
    sorted_entries = sorted(entries, key=lambda item: item[1], reverse=True)
    return sorted_entries[:k]


def compute_predictions(engine: KGEInference, top_k: int) -> Dict[str, float]:
    """Compute top-k predictions for each predicate/constant role combination."""
    best_scores: Dict[str, float] = {}
    scores_cache: Dict[Tuple[str, str, str], float] = {}

    for predicate in engine.fol.predicates:
        if predicate.arity != 2:
            continue

        predicate_name = predicate.name
        head_constants = predicate.domains[0].constants
        tail_constants = predicate.domains[1].constants

        if not head_constants or not tail_constants:
            continue

        # Iterate by fixing the head constant and ranking tail candidates.
        for head_const in head_constants:
            candidate_atoms = [(predicate_name, head_const, tail_const) for tail_const in tail_constants]
            scores = engine.predict_batch(candidate_atoms)

            for atom, score in zip(candidate_atoms, scores):
                scores_cache[atom] = score

            head_rankings = rank_top_k(zip(candidate_atoms, scores), top_k)
            for atom, score in head_rankings:
                fact_string = f"{atom[0]}({atom[1]},{atom[2]})"
                best_scores[fact_string] = max(score, best_scores.get(fact_string, float('-inf')))

        # Iterate by fixing the tail constant and ranking head candidates.
        for tail_const in tail_constants:
            tail_candidates: List[Tuple[str, float]] = []
            for head_const in head_constants:
                atom = (predicate_name, head_const, tail_const)
                if atom not in scores_cache:
                    atom_string = f"{predicate_name}({head_const},{tail_const})"
                    scores_cache[atom] = engine.predict(atom_string)
                tail_candidates.append((head_const, scores_cache[atom]))

            tail_rankings = rank_top_k(((predicate_name, head_const, tail_const), score)
                                       for head_const, score in tail_candidates)
            for atom, score in tail_rankings:
                fact_string = f"{atom[0]}({atom[1]},{atom[2]})"
                best_scores[fact_string] = max(score, best_scores.get(fact_string, float('-inf')))

    return best_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-k KGE predictions for each relation and constant role.")
    parser.add_argument("--dataset", default="family", help="Dataset name (e.g., countries_s3, family, wn18rr).")
    parser.add_argument("--data-path", default=os.path.join(current_dir, "data"),
                        help="Base path containing dataset folders (default: experiments/data).")
    parser.add_argument("--checkpoint-dir", default=os.path.join("..", "checkpoints"),
                        help="Directory with KGE checkpoints (default: ../checkpoints).")
    parser.add_argument("--run-signature", default=None, help="Run signature identifying the checkpoint to load.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when training the checkpoint (default: 0).")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top predictions to keep for each role (default: 2).")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: experiments/runs/<dataset>_kge_topk.txt).")
    args = parser.parse_args()

    run_signature = args.run_signature or infer_default_signature(args.dataset)

    output_path = args.output
    if output_path is None:
        output_filename = f"{args.dataset}_kge_top{args.top_k}.txt"
        output_path = os.path.join(current_dir, "runs", output_filename)

    ensure_parent_dir(output_path)
    print('data path:', args.data_path)
    engine = KGEInference(
        dataset_name=args.dataset,
        base_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        run_signature=run_signature,
        seed=args.seed,
    )

    best_scores = compute_predictions(engine, args.top_k)

    with open(output_path, "w", encoding="ascii") as fh:
        for fact in sorted(best_scores.keys()):
            fh.write(f"{fact} {best_scores[fact]:.6f}\n")

    print(f"Stored {len(best_scores)} facts in {output_path}")


if __name__ == "__main__":
    main()
