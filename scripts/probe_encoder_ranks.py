from __future__ import annotations

import argparse

from marginal_value.models.ts2vec_diagnostics import run_layer_rank_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default="cache/manifests/pretrain_full_cached_urls.txt")
    parser.add_argument("--n-clips", type=int, default=32)
    parser.add_argument("--clip-len", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--output-dims", type=int, default=320)
    parser.add_argument("--output-path")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--first-window", action="store_true")
    args = parser.parse_args()

    results = run_layer_rank_probe(
        manifest_path=args.manifest_path,
        n_clips=args.n_clips,
        clip_len=args.clip_len,
        device=args.device,
        hidden_dims=args.hidden_dims,
        output_dims=args.output_dims,
        output_path=args.output_path,
        seed=args.seed,
        random_start=not args.first_window,
    )
    for row in results:
        print(
            f"{row['layer']:>22}: "
            f"rank={row['effective_rank']:7.4f}  "
            f"cosine={row['mean_pairwise_cosine']:7.4f}  "
            f"std={row['activation_std']:8.6f}  "
            f"mean_abs={row['activation_mean_abs']:8.6f}"
        )


if __name__ == "__main__":
    main()
