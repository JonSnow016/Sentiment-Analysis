"""
02_wordclouds.py

Generate word clouds from the long format dataset.

Outputs are written under outputs/WordCloud/.
"""

from __future__ import annotations

import argparse

from imports import Path, WordCloud, plt, pd, get_stop_words, safe_name


def make_wc(text: str, out_path: Path, title: str | None = None) -> None:
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        stopwords=set(get_stop_words()),
        collocations=False,
        regexp=r"[A-Za-z]{3,}",
    ).generate(str(text))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(out_path))

    # Optional display (kept lightweight; safe to remove if running headless)
    if title:
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/df_posts_long.xlsx",
        help="Path to the long format dataset created by 01_build_long_dataset.py.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/WordCloud",
        help="Output directory.",
    )
    parser.add_argument(
        "--by_candidate",
        action="store_true",
        help="Also generate word clouds for each candidate within each state.",
    )
    args = parser.parse_args()

    df_path = Path(args.data)
    out_dir = Path(args.out_dir)

    if not df_path.exists():
        raise FileNotFoundError(
            f"Long dataset not found: {df_path}. Run 01_build_long_dataset.py first."
        )

    df = pd.read_excel(df_path, dtype=str)
    df["post_text"] = df["post_text"].astype(str)
    df["state"] = df["state"].astype(str)
    df["candidate_header"] = df["candidate_header"].astype(str)

    state_docs = (
        df.groupby("state")["post_text"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index(name="doc")
    )

    state_folder = out_dir / "State"
    state_folder.mkdir(parents=True, exist_ok=True)

    for _, row in state_docs.iterrows():
        st = row["state"]
        text = row["doc"]
        out_path = state_folder / f"wordcloud__{safe_name(st)}.png"
        make_wc(text, out_path, title=f"{st} | All candidates")

    if args.by_candidate:
        cand_docs = (
            df.groupby(["state", "candidate_header"])["post_text"]
            .apply(lambda s: " ".join(s.astype(str)))
            .reset_index(name="doc")
        )

        cand_folder = out_dir / "Candidate"
        for _, row in cand_docs.iterrows():
            st = row["state"]
            cand = row["candidate_header"]
            text = row["doc"]
            st_dir = cand_folder / safe_name(st)
            out_path = st_dir / f"wordcloud__{safe_name(cand)}.png"
            make_wc(text, out_path, title=f"{st} | {cand}")

    print(f"Saved word clouds under: {out_dir}")


if __name__ == "__main__":
    main()
