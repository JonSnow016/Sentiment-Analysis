"""
03_frequency_and_zipf.py

Reproduce the frequency based outputs from the notebook:
1) Candidate polar bar charts of top words
2) 3D candidate versus top word frequency plots by state
3) Candidate level histograms of top words
4) Zipf distribution plots and slope estimates

Outputs are written under outputs/ using folder names aligned with the Google Drive structure.
"""

from __future__ import annotations

import argparse
from math import pi

from imports import (
    CountVectorizer,
    FixedFormatter,
    FixedLocator,
    Path,
    TOKEN_PATTERN,
    get_stop_words,
    np,
    pd,
    plt,
    safe_name,
)


def build_vectorizer():
    return CountVectorizer(
        stop_words=get_stop_words(),
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
    )


def short(s: str, n: int = 14) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def candidate_docs_by_state(df: pd.DataFrame, state: str) -> pd.DataFrame:
    sub = df[df["state"] == state]
    return (
        sub.groupby("candidate_header", sort=False)["post_text"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index(name="doc")
    )


def polar_bar_candidate_top_words(df: pd.DataFrame, out_dir: Path, top_words: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for state in sorted(df["state"].unique()):
        cand_docs = candidate_docs_by_state(df, state)
        if cand_docs.empty:
            continue

        vec = build_vectorizer()
        X = vec.fit_transform(cand_docs["doc"])
        vocab = vec.get_feature_names_out()

        if X.shape[1] == 0:
            continue

        totals = np.asarray(X.sum(axis=0)).ravel()
        top_idx = totals.argsort()[::-1][:top_words]
        words = vocab[top_idx].tolist()

        Z = X[:, top_idx].toarray()

        thetas = np.linspace(0.0, 2 * pi, len(words), endpoint=False)
        width = (2 * pi / max(1, len(words))) * 0.85

        for i, cand in enumerate(cand_docs["candidate_header"].tolist()):
            vals = Z[i].astype(float)

            fig = plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection="polar")

            ax.bar(thetas, vals, width=width, alpha=0.75)
            ax.set_xticks(thetas)
            ax.set_xticklabels(words, fontweight="bold")
            ax.set_ylim(0, max(1.0, float(vals.max())) * 1.20)

            ax.set_title(
                f"{state} | {cand} | Top {top_words} Words",
                fontweight="bold",
                va="bottom",
                pad=25,
            )

            plt.tight_layout()

            fname = f"polarbar_{state}_{i+1}.png".replace("/", "_")
            plt.savefig(out_dir / fname, dpi=250, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)


def frequency_3d_plots(df: pd.DataFrame, out_dir: Path, top_words: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for state in sorted(df["state"].unique()):
        cand_docs = candidate_docs_by_state(df, state)
        if cand_docs.empty:
            continue

        vec = build_vectorizer()
        X = vec.fit_transform(cand_docs["doc"])
        vocab = vec.get_feature_names_out()

        if X.shape[1] == 0:
            print(f"Skipping {state}: no vocabulary after filtering.")
            continue

        totals = np.asarray(X.sum(axis=0)).ravel()
        top_idx = totals.argsort()[::-1][:top_words]
        top_words_full = vocab[top_idx]
        top_words_short = [short(w, 14) for w in top_words_full]

        Z = X[:, top_idx].toarray()
        n_cand, n_words = Z.shape

        x = np.tile(np.arange(n_words), n_cand).astype(float)
        y = np.repeat(np.arange(n_cand), n_words).astype(float)
        dz = Z.ravel().astype(float)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.bar3d(x, y, np.zeros_like(dz), 1.0, 1.0, dz, shade=True)

        ax.set_xlim(0, n_words)
        ax.set_ylim(0, n_cand)

        xcent = np.arange(n_words) + 0.5
        ycent = np.arange(n_cand) + 0.7

        ax.xaxis.set_major_locator(FixedLocator(xcent))
        ax.xaxis.set_major_formatter(FixedFormatter([short(w, 12) for w in top_words_short]))

        ax.yaxis.set_major_locator(FixedLocator(ycent))
        ax.yaxis.set_major_formatter(FixedFormatter([short(c, 28) for c in cand_docs["candidate_header"].tolist()]))

        for lab in ax.get_xticklabels():
            lab.set_rotation(25)
            lab.set_ha("right")

        for lab in ax.get_yticklabels():
            lab.set_rotation(0)
            lab.set_ha("left")
            lab.set_va("center")

        ax.set_title(f"{state} | Candidate vs Top Words (Frequency)")
        ax.set_zlabel("Frequency", labelpad=40)
        ax.view_init(elev=40, azim=-40)

        plt.subplots_adjust(left=0.06, right=0.86, bottom=0.18, top=0.90)
        plt.savefig(out_dir / f"3d_{state}.png", dpi=250, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def candidate_histograms(df: pd.DataFrame, out_dir: Path, top_words: int = 10) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for state in sorted(df["state"].unique()):
        state_dir = out_dir / safe_name(state)
        state_dir.mkdir(parents=True, exist_ok=True)

        cand_docs = candidate_docs_by_state(df, state)
        for _, row in cand_docs.iterrows():
            candidate = row["candidate_header"]
            text = row["doc"]

            vec = build_vectorizer()
            X = vec.fit_transform([text])
            vocab = vec.get_feature_names_out()
            counts = X.toarray().ravel()

            if len(vocab) == 0:
                continue

            idx = counts.argsort()[::-1][:top_words]
            words = vocab[idx]
            freqs = counts[idx]

            fig = plt.figure(figsize=(7, 4))
            plt.bar(words, freqs)
            plt.title(f"{candidate} | {state} | Top {top_words} Words", fontweight="bold")
            plt.xlabel("Words", fontweight="bold")
            plt.ylabel("Frequency", fontweight="bold")
            plt.xticks(rotation=40, ha="right")
            plt.tight_layout()

            fname = f"hist_{safe_name(candidate)}.png"
            plt.savefig(state_dir / fname, dpi=250, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)


def zipf_all_posts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_text = " ".join(df["post_text"].astype(str).tolist())
    vec = build_vectorizer()
    X = vec.fit_transform([all_text])
    vocab = vec.get_feature_names_out()

    counts = np.asarray(X.sum(axis=0)).ravel().astype(int)
    mask = counts > 0

    wf = pd.DataFrame({"word": vocab[mask], "frequency": counts[mask]})
    wf = wf.sort_values("frequency", ascending=False).reset_index(drop=True)

    rank = np.arange(1, len(wf) + 1)
    freq = wf["frequency"].to_numpy()

    fig = plt.figure(figsize=(7, 5))
    plt.loglog(rank, freq, marker=".", linestyle="none")
    plt.title("Zipf plot (all posts combined)")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.tight_layout()
    plt.savefig(out_dir / "zipf_plot.png", dpi=250, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    rmin, rmax = 20, min(500, len(rank))
    r = rank[rmin - 1 : rmax]
    f = freq[rmin - 1 : rmax]

    if len(r) > 1:
        b, a = np.polyfit(np.log10(r), np.log10(f), 1)
    else:
        b, a = np.nan, np.nan

    fig = plt.figure(figsize=(7, 5))
    plt.loglog(rank, freq, marker=".", linestyle="none")
    if len(r) > 1:
        plt.loglog(r, 10 ** (a + b * np.log10(r)), linewidth=2, label=f"slope = {b:.2f} (ranks {rmin} to {rmax})")
    plt.title("Zipf plot with slope fit")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    if len(r) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "zipf_plot_with_slope.png", dpi=250, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    # Excel export
    excel_1 = out_dir / "word_frequencies.xlsx"
    excel_2 = out_dir / "word_frequencies (1).xlsx"
    with pd.ExcelWriter(excel_1, engine="openpyxl") as writer:
        wf.to_excel(writer, sheet_name="AllPosts_WordFreq", index=False)
        wf.head(10).to_excel(writer, sheet_name="Top10_Preview", index=False)
    # Duplicate name aligned with the Google Drive export
    try:
        excel_2.write_bytes(excel_1.read_bytes())
    except Exception:
        pass


def zipf_by_state(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def zipf_arrays(text: str):
        vec = build_vectorizer()
        X = vec.fit_transform([text])
        counts = np.asarray(X.sum(axis=0)).ravel().astype(int)
        counts = counts[counts > 0]
        counts = np.sort(counts)[::-1]
        rank = np.arange(1, len(counts) + 1)
        return rank, counts

    fig = plt.figure(figsize=(7, 5))
    for state in sorted(df["state"].unique()):
        text_state = " ".join(df.loc[df["state"] == state, "post_text"].astype(str).tolist())
        rank, freq = zipf_arrays(text_state)
        plt.loglog(rank, freq, marker=".", linestyle="none", alpha=0.65, label=state)

    plt.title("Zipf Distribution by State (All Posts)")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ZipF Distribution States.png", dpi=250, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    rmin = 20
    slopes = []
    with pd.ExcelWriter(out_dir / "word_frequencies_by_state.xlsx", engine="openpyxl") as writer:
        for state in sorted(df["state"].unique()):
            text_state = " ".join(df.loc[df["state"] == state, "post_text"].astype(str).tolist())
            rank, freq = zipf_arrays(text_state)

            rmax = min(500, len(rank))
            if rmax <= rmin:
                slopes.append([state, np.nan, rmin, rmax])
            else:
                r = rank[rmin - 1 : rmax]
                f = freq[rmin - 1 : rmax]
                b, a = np.polyfit(np.log10(r), np.log10(f), 1)
                slopes.append([state, b, rmin, rmax])

            # per state word frequencies
            vec = build_vectorizer()
            X = vec.fit_transform([text_state])
            vocab = vec.get_feature_names_out()
            counts = np.asarray(X.sum(axis=0)).ravel().astype(int)
            mask = counts > 0
            wf_state = pd.DataFrame({"word": vocab[mask], "frequency": counts[mask]}).sort_values(
                "frequency", ascending=False
            )
            sheet = str(state)[:31]
            wf_state.to_excel(writer, sheet_name=sheet, index=False)

    slope_df = pd.DataFrame(slopes, columns=["state", "zipf_slope", "rmin", "rmax"])
    slope_df.to_excel(out_dir / "zipf_state_slopes.xlsx", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/df_posts_long.xlsx",
        help="Path to the long format dataset created by 01_build_long_dataset.py.",
    )
    parser.add_argument(
        "--top_words",
        type=int,
        default=5,
        help="Number of top words used in polar bar and 3D frequency plots.",
    )
    args = parser.parse_args()

    df_path = Path(args.data)
    if not df_path.exists():
        raise FileNotFoundError(
            f"Long dataset not found: {df_path}. Run 01_build_long_dataset.py first."
        )

    df = pd.read_excel(df_path, dtype=str)
    df["post_text"] = df["post_text"].astype(str)
    df["state"] = df["state"].astype(str)
    df["candidate_header"] = df["candidate_header"].astype(str)

    outputs_root = Path("outputs")

    polar_dir = outputs_root / "PolarPlot Candidate"
    freq_dir = outputs_root / "FrequencyPlots"
    hist_dir = outputs_root / "Histogram"
    zipf_dir = outputs_root / "ZipfDistribution"

    polar_bar_candidate_top_words(df, polar_dir, top_words=args.top_words)
    frequency_3d_plots(df, freq_dir, top_words=args.top_words)
    candidate_histograms(df, hist_dir, top_words=10)
    zipf_all_posts(df, zipf_dir)
    zipf_by_state(df, zipf_dir)

    print("Saved outputs under outputs/.")


if __name__ == "__main__":
    main()
