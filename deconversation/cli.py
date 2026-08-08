import argparse
import sys
import os

def build_parser():
    parser = argparse.ArgumentParser(
        prog="deconverse",
        description=(
            "Extract LLM-derived embeddings for bulk and reference "
            "signature data, then deconvolve bulk RNA-seq into cell "
            "type proportions."
        ),
    )
    parser.add_argument(
        "-b", "--bulk",
        required=True,
        help="Path to bulk expression matrix CSV (rows: genes, columns: samples)",
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to zero-shot or fine-tuned scFM model",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["geneformer", "c2s", "cellhermes", "scgpt"],
        help="scFM model type (inferred from model path if omitted)",
    )
    parser.add_argument(
        "-a", "--adata",
        default=None,
        help="Path to reference single-cell .h5ad object (required if --sig-df not given)",
    )
    parser.add_argument(
        "-s", "--sig",
        default=None,
        help="Path to precomputed signature matrix CSV",
    )
    parser.add_argument(
        "-d", "--temp-output-dir",
        default="temp",
        help="Directory for saving intermediate files (default: temp)",
    )
    parser.add_argument(
        "--cell-type-col",
        default="type",
        help="adata.obs column designating cell type (default: type)",
    )
    parser.add_argument(
        "--sample-col",
        default="sample",
        help="adata.obs column designating sample or batch (default: sample)",
    )
    parser.add_argument(
        "--solver",
        default="nnls",
        choices=["nnls", "dwls", "ridge", "elasticnet", "nusvr"],
        help="Deconvolution solver (default: nnls)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to write output CSV of cell-type proportions "
             "(defaults to stdout as CSV if omitted)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.adata is None and args.sig is None:
        parser.error("either --adata or --sig must be provided")

    os.makedirs(args.temp_output_dir, exist_ok=True)
    
    from .core import deconverse
    try:
        result = deconverse(
            bulk_df=args.bulk,
            model=args.model,
            mode=args.mode,
            adata=args.adata,
            sig_df=args.sig,
            temp_output_dir=args.temp_output_dir,
            cell_type_col=args.cell_type_col,
            sample_col=args.sample_col,
            solver=args.solver,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        result.to_csv(args.output)
        print(f"Saved cell-type proportions to {args.output}")
    else:
        print(result.to_csv())


if __name__ == "__main__":
    main()
