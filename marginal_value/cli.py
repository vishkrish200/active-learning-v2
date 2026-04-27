from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from marginal_value.data.load_imu import load_imu_directory
from marginal_value.indexing.cluster_features import new_batch_support_features
from marginal_value.indexing.knn_features import ExactKNNIndex, build_old_support_features
from marginal_value.models.encoder import HandcraftedIMUEncoder
from marginal_value.models.ranker import score_candidates
from marginal_value.preprocessing.quality import compute_quality_features
from marginal_value.submit.make_submission import build_submission_rows, diversity_rerank, write_diagnostics
from marginal_value.training.config import (
    LocalTrainingDisabledError,
    build_modal_run_command,
    load_training_config,
    refuse_local_training,
    validate_training_dispatch,
)
from marginal_value.ranking.audit_submission import build_ranking_model_card, write_ranking_model_card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marginal-value")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank_parser = subparsers.add_parser("rank", help="Rank new worker IMU CSV clips")
    rank_parser.add_argument("--existing-dir", required=True)
    rank_parser.add_argument("--new-dir", required=True)
    rank_parser.add_argument("--submission-out", required=True)
    rank_parser.add_argument("--diagnostics-out", required=True)
    rank_parser.add_argument("--sample-rate", type=float, default=30.0)
    rank_parser.add_argument("--lambda-redundancy", type=float, default=0.25)

    validate_parser = subparsers.add_parser(
        "validate-training",
        help="Validate Modal training config locally without running training",
    )
    validate_parser.add_argument("--config", default="configs/modal_training.json")

    train_parser = subparsers.add_parser(
        "train",
        help="Refuse local training and print the Modal command to use instead",
    )
    train_parser.add_argument("--config", default="configs/modal_training.json")
    train_parser.add_argument("--validation", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit-submission",
        help="Build a lightweight ranking model-card audit from submission diagnostics",
    )
    audit_parser.add_argument("--submission", required=True)
    audit_parser.add_argument("--diagnostics", required=True)
    audit_parser.add_argument("--out", required=True)
    audit_parser.add_argument("--candidate-scores")
    audit_parser.add_argument("--quality-metadata")
    audit_parser.add_argument("--run-report")
    audit_parser.add_argument("--config")
    audit_parser.add_argument("--run-name")
    audit_parser.add_argument("--top-k", nargs="+", type=int, default=[10, 25, 50, 100, 200])
    audit_parser.add_argument("--low-quality-threshold", type=float, default=0.45)

    args = parser.parse_args(argv)
    if args.command == "rank":
        return _rank(args)
    if args.command == "validate-training":
        config = load_training_config(args.config)
        validate_training_dispatch(config)
        print("Training config is valid for Modal H100 dispatch. No local training was run.")
        return 0
    if args.command == "train":
        command = " ".join(
            build_modal_run_command(
                args.config,
                run_validation=args.validation,
                run_full=False,
            )
        )
        print(f"Use Modal for training: {command}")
        try:
            refuse_local_training()
        except LocalTrainingDisabledError as error:
            print(error)
            return 2
    if args.command == "audit-submission":
        card = build_ranking_model_card(
            submission_path=args.submission,
            diagnostics_path=args.diagnostics,
            candidate_path=args.candidate_scores,
            quality_metadata_path=args.quality_metadata,
            run_report_path=args.run_report,
            config_path=args.config,
            run_name=args.run_name,
            top_ks=args.top_k,
            low_quality_threshold=args.low_quality_threshold,
        )
        output_path = write_ranking_model_card(card, args.out)
        print(f"Wrote ranking model card: {output_path}")
        return 0
    return 1


def _rank(args: argparse.Namespace) -> int:
    old_workers = load_imu_directory(args.existing_dir)
    new_workers = load_imu_directory(args.new_dir)

    encoder = HandcraftedIMUEncoder(sample_rate=args.sample_rate)
    old_embeddings = encoder.encode_many([worker.samples for worker in old_workers])
    new_embeddings = encoder.encode_many([worker.samples for worker in new_workers])

    old_index = ExactKNNIndex().fit(old_embeddings)
    old_support = build_old_support_features(old_index, new_embeddings)
    new_support = new_batch_support_features(new_embeddings)

    feature_rows = []
    for worker, old_row, new_row in zip(new_workers, old_support, new_support):
        quality = compute_quality_features(
            worker.samples,
            timestamps=worker.timestamps,
            sample_rate=args.sample_rate,
        )
        row = {
            "worker_id": worker.worker_id,
            **old_row,
            **new_row,
            **quality,
            "token_nll_mean": 0.0,
            "token_nll_p95": 0.0,
            "rare_phrase_fraction": 0.0,
        }
        feature_rows.append(row)

    scored = score_candidates(feature_rows)
    reranked = diversity_rerank(scored, new_embeddings, lambda_redundancy=args.lambda_redundancy)
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = rank

    submission = build_submission_rows(reranked)
    submission_path = Path(args.submission_out)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(submission).to_csv(submission_path, index=False)
    write_diagnostics(reranked, args.diagnostics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
