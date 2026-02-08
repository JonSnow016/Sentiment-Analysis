"""
Build a tidy, long format dataset from the Excel workbook.

Input format:
Candidate headers are in row 1 of columns A, C, E, ...
Posts for that candidate are in the adjacent column B, D, F, ... down the rows.
"""

from __future__ import annotations

import argparse

from imports import pd, Path, project_paths


def reshape_paired_columns(file_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)

    out: list[dict] = []
    ncols = raw.shape[1]

    for c in range(0, ncols, 2):
        if c + 1 >= ncols:
            break

        cand = raw.iat[0, c]
        if cand is None:
            continue

        cand = str(cand).strip()
        if not cand:
            continue

        posts = raw.iloc[1:, c + 1].dropna()
        for r_i, post in posts.items():
            post = str(post).strip()
            if not post:
                continue

            out.append(
                {
                    "state": sheet_name,
                    "candidate_header": cand,
                    "post_text": post,
                    "post_order": int(r_i),
                }
            )

    return pd.DataFrame(out)


def build_long_dataset(file_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    frames = [reshape_paired_columns(file_path, st) for st in sheets]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df


def main() -> None:
    paths = project_paths()
    default_file = paths["data"] / "FemaleCandidatesData.xlsx"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_file),
        help="Path to the Excel workbook containing the scraped posts.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(paths["outputs"] / "df_posts_long.xlsx"),
        help="Path to save the long format dataset.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. Place the workbook under data/ or pass --input."
        )

    df = build_long_dataset(in_path)
    df.to_excel(out_path, index=False)

    print(f"Saved long dataset: {out_path}")
    print(f"Total posts: {len(df)}")
    if not df.empty:
        print("Posts per state:")
        print(df.groupby('state').size().sort_values(ascending=False))
        print("Candidates per state:")
        print(df.groupby('state')['candidate_header'].nunique().sort_values(ascending=False))


if __name__ == "__main__":
    main()
