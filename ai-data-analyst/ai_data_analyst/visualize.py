"""Automatic chart generation from query results (matplotlib, headless)."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PALETTE = ["#2E6FD8", "#E8833A", "#2FA37A", "#B456B5", "#D9534F", "#7A7F87", "#C9A227", "#3AA8C1"]
VALID_TYPES = {"bar", "line", "pie", "scatter", "hist", "none"}
_TIME_RE = re.compile(r"^\d{4}(-|$)")


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _looks_temporal(s: pd.Series) -> bool:
    return not s.empty and not _is_numeric(s) and bool(_TIME_RE.match(str(s.iloc[0])))


def infer_spec(df: pd.DataFrame) -> dict:
    """Pick a sensible chart when the model gave none (or gave an invalid one)."""
    if df.empty or len(df.columns) < 1:
        return {"type": "none"}
    nums = [c for c in df.columns if _is_numeric(df[c])]
    cats = [c for c in df.columns if c not in nums]
    if len(df) == 1:
        return {"type": "none"}
    if len(cats) >= 2 and nums and _looks_temporal(df[cats[0]]):
        return {"type": "line", "x": cats[0], "series": cats[1], "y": nums[-1]}
    if cats and nums:
        kind = "line" if _looks_temporal(df[cats[0]]) else "bar"
        return {"type": kind, "x": cats[0], "y": nums[-1]}
    if len(nums) >= 2:
        return {"type": "scatter", "x": nums[0], "y": nums[1]}
    if len(nums) == 1:
        return {"type": "hist", "x": nums[0]}
    return {"type": "none"}


def _valid(spec: dict, df: pd.DataFrame) -> bool:
    kind = spec.get("type")
    if kind not in VALID_TYPES:
        return False
    if kind == "none":
        return True
    cols = set(df.columns)
    if spec.get("x") not in cols:
        return False
    if spec.get("series") and spec["series"] not in cols:
        return False
    if kind == "hist":
        return True
    ys = _ys(spec)
    if not ys or any(y not in cols for y in ys):
        return False
    if len(ys) > 1 and (kind not in {"bar", "line"} or spec.get("series")):
        return False
    return all(_is_numeric(df[y]) for y in ys)


def _ys(spec: dict) -> list:
    """`y` may be one column name or a list of columns (several measures on one axis)."""
    y = spec.get("y")
    if y is None:
        return []
    return list(y) if isinstance(y, (list, tuple)) else [y]


def _label(name: str) -> str:
    return str(name).replace("_", " ").title()


def make_chart(df: pd.DataFrame, spec: dict | None, out_path: Path, question: str = "") -> Path | None:
    """Render a chart for `df` to `out_path` (PNG). Returns the path, or None if no chart fits."""
    spec = dict(spec or {})
    if not _valid(spec, df):
        spec = infer_spec(df)
    kind = spec.get("type", "none")
    if kind == "none" or df.empty:
        return None

    title = spec.get("title") or question or "Result"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    x, series = spec.get("x"), spec.get("series")
    ys = _ys(spec)
    y = ys[0] if ys else None

    if kind == "line":
        if series:
            pivot = df.pivot_table(index=x, columns=series, values=y, aggfunc="sum").sort_index()
            for i, col in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[col], marker="o", ms=3, lw=2, color=PALETTE[i % len(PALETTE)], label=str(col))
            ax.legend(title=_label(series), frameon=False)
        else:
            for i, col in enumerate(ys):
                ax.plot(df[x], df[col], marker="o", ms=4, lw=2, color=PALETTE[i % len(PALETTE)], label=_label(col))
            if len(ys) > 1:
                ax.legend(frameon=False)
        _thin_ticks(ax, df[x].nunique())
    elif kind == "bar":
        data = df.head(25)
        labels = data[x].astype(str)
        width = 0.8 / len(ys)
        for i, col in enumerate(ys):
            offset = (i - (len(ys) - 1) / 2) * width
            ax.bar([p + offset for p in range(len(data))], data[col], width=width,
                   color=PALETTE[i % len(PALETTE)], label=_label(col))
        ax.set_xticks(range(len(data)), labels)
        if len(ys) > 1:
            ax.legend(frameon=False)
        ax.tick_params(axis="x", rotation=35 if len(data) > 5 else 0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right" if len(data) > 5 else "center")
    elif kind == "pie":
        data = df.head(8)
        ax.pie(data[y], labels=data[x].astype(str), autopct="%1.1f%%", colors=PALETTE, startangle=90)
        ax.axis("equal")
    elif kind == "scatter":
        ax.scatter(df[x], df[y], s=18, alpha=0.7, color=PALETTE[0])
    elif kind == "hist":
        ax.hist(df[x].dropna(), bins=min(30, max(5, len(df) // 10)), color=PALETTE[0], edgecolor="white")

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    if kind != "pie":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:,.2f}"))
        ax.set_xlabel(_label(x))
        ax.set_ylabel(_label(y) if len(ys) == 1 else ("Value" if ys else "Count"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _thin_ticks(ax, n_points: int) -> None:
    """Avoid crowded x labels on long time series."""
    if n_points > 12:
        step = max(1, n_points // 10)
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_visible(i % step == 0)
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
