#!/usr/bin/env python3
"""Plot autoresearch progress from results.tsv as a standalone SVG or PNG."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path


@dataclass
class Result:
    cycle: int
    timestamp: datetime
    commit: str
    accepted: bool
    metric: float
    is_valid: bool
    description: str


@dataclass
class Label:
    row: Result
    text: str
    anchor: str


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_results(path: Path, metric: str) -> list[Result]:
    rows: list[Result] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"{path} is empty")
        missing = {metric, "cycle", "timestamp", "commit", "accepted", "is_valid"} - set(
            reader.fieldnames
        )
        if missing:
            raise SystemExit(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        for raw in reader:
            try:
                rows.append(
                    Result(
                        cycle=int(raw["cycle"]),
                        timestamp=parse_timestamp(raw["timestamp"]),
                        commit=raw["commit"],
                        accepted=raw["accepted"] == "1",
                        metric=float(raw[metric]),
                        is_valid=raw["is_valid"] == "1",
                        description=raw.get("commit_description", "").strip(),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"could not parse row {raw!r}: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path} has no result rows")
    return rows


def best_steps(rows: list[Result], lower_is_better: bool) -> list[Result]:
    best: Result | None = None
    steps: list[Result] = []
    for row in rows:
        if not row.accepted:
            continue
        if best is None:
            best = row
            steps.append(row)
            continue
        improved = row.metric < best.metric if lower_is_better else row.metric > best.metric
        if improved:
            best = row
            steps.append(row)
    return steps


def metric_gain(previous: Result, current: Result, lower_is_better: bool) -> float:
    if lower_is_better:
        return previous.metric - current.metric
    return current.metric - previous.metric


def select_improvement_labels(
    improvements: list[Result],
    *,
    lower_is_better: bool,
    major_min_gain: float,
    large_gain_fraction: float,
    huge_gain_fraction: float,
    quiet_cycles: int,
    quiet_hours: float,
    min_label_cycles: int,
    min_label_hours: float,
    max_labels: int,
) -> list[Label]:
    if not improvements:
        return []

    labels: list[Label] = [Label(improvements[0], label_for(improvements[0], True), "start")]
    if len(improvements) == 1:
        return labels

    gains = [metric_gain(prev, cur, lower_is_better) for prev, cur in zip(improvements, improvements[1:])]
    largest_gain = max(gains) if gains else 0.0
    large_gain = largest_gain * large_gain_fraction
    huge_gain = largest_gain * huge_gain_fraction
    last_labeled = improvements[0]
    label_rows = {improvements[0].cycle}

    for index, row in enumerate(improvements[1:], start=1):
        gain = gains[index - 1]
        cycle_gap = row.cycle - last_labeled.cycle
        hour_gap = (row.timestamp - last_labeled.timestamp).total_seconds() / 3600.0
        is_last = index == len(improvements) - 1
        spaced = cycle_gap >= min_label_cycles or hour_gap >= min_label_hours
        worth_labeling = (
            (major_min_gain > 0 and gain >= major_min_gain)
            or gain >= huge_gain
            or (gain >= large_gain and spaced)
            or cycle_gap >= quiet_cycles
            or hour_gap >= quiet_hours
            or is_last
        )
        if worth_labeling and row.cycle not in label_rows:
            label_rows.add(row.cycle)
            labels.append(Label(row, label_for(row, False), "improvement"))
            last_labeled = row

    if max_labels > 0 and len(labels) > max_labels:
        first = labels[0]
        last = labels[-1]
        middle = labels[1:-1]
        if max_labels <= 2:
            labels = [first, last][:max_labels]
        else:
            step = max(1, len(middle) // (max_labels - 2))
            labels = [first] + middle[::step][: max_labels - 2] + [last]

    return labels


def label_for(row: Result, first: bool) -> str:
    if first:
        return "baseline"
    text = row.description
    prefix = f"cycle {row.cycle}:"
    if text.lower().startswith(prefix):
        text = text[len(prefix) :].strip()
    if not text:
        text = row.commit[:8]
    if len(text) > 42:
        text = text[:39].rstrip() + "..."
    return text


def nice_number(value: float) -> str:
    if abs(value) >= 100 or abs(value) < 0.001:
        return f"{value:.3g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def render_svg(
    rows: list[Result],
    metric: str,
    output: Path,
    *,
    title: str | None,
    lower_is_better: bool,
    max_labels: int,
    major_min_gain: float,
    large_gain_fraction: float,
    huge_gain_fraction: float,
    quiet_cycles: int,
    quiet_hours: float,
    min_label_cycles: int,
    min_label_hours: float,
    width: int,
    height: int,
    experiment_count: int | None = None,
) -> None:
    margin_left = 110
    margin_right = 50
    margin_top = 70
    margin_bottom = 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    times = [row.timestamp.timestamp() for row in rows]
    values = [row.metric for row in rows]
    min_x, max_x = min(times), max(times)
    min_y, max_y = min(values), max(values)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        pad = abs(min_y) * 0.01 or 0.01
        min_y -= pad
        max_y += pad
    y_pad = (max_y - min_y) * 0.08
    min_y -= y_pad
    max_y += y_pad

    def x_pos(ts: float) -> float:
        return margin_left + (ts - min_x) / (max_x - min_x) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + (max_y - value) / (max_y - min_y) * plot_h

    accepted = [row for row in rows if row.accepted]
    improvements = best_steps(rows, lower_is_better)
    improvement_count = max(0, len(improvements) - 1)
    final_row = rows[-1]
    best_row = improvements[-1] if improvements else final_row
    total = experiment_count if experiment_count is not None else len(rows)
    heading = title or f"Autoresearch Progress: {total} Experiments, {improvement_count} Kept Improvements"
    direction = "lower is better" if lower_is_better else "higher is better"

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append("<style>")
    parts.append(
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#1f2933}"
        ".grid{stroke:#e6e6e6;stroke-width:1}"
        ".axis{stroke:#222;stroke-width:1.3}"
        ".discarded{fill:#b9b9b9;opacity:.42}"
        ".kept{fill:#32c77a;stroke:#0b6b3a;stroke-width:1.5}"
        ".best{fill:none;stroke:#63c88c;stroke-width:3}"
        ".label{fill:#2d9557;font-size:14px;font-weight:500}"
        ".label-line{stroke:#2d9557;stroke-width:1;opacity:.55}"
        ".note{fill:#334155;font-size:15px}"
        ".note-strong{fill:#0f5132;font-size:17px;font-weight:650}"
        ".tick{font-size:13px;fill:#333}"
        ".legend{font-size:15px}"
    )
    parts.append("</style>")
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        f'font-size="26">{escape(heading)}</text>'
    )
    parts.append(
        f'<text x="{width / 2:.1f}" y="58" text-anchor="middle" class="note">'
        f"Best {escape(metric)}: {best_row.metric:.6f} at cycle {best_row.cycle}; "
        f"latest recorded: {final_row.metric:.6f} at cycle {final_row.cycle}</text>"
    )

    # Grid and y-axis ticks.
    tick_count = 6
    for idx in range(tick_count + 1):
        y_value = min_y + (max_y - min_y) * idx / tick_count
        y = y_pos(y_value)
        parts.append(
            f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.1f}" '
            f'text-anchor="end">{escape(nice_number(y_value))}</text>'
        )

    # Time ticks.
    for idx in range(7):
        ts = min_x + (max_x - min_x) * idx / 6
        x = x_pos(ts)
        label = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" '
            f'x2="{x:.1f}" y2="{height - margin_bottom}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - margin_bottom + 24}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{height - margin_bottom}"/>'
    )
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 34}" text-anchor="middle" '
        f'font-size="18">Time</text>'
    )
    parts.append(
        f'<text transform="translate(28 {height / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-size="18">{escape(metric)} ({direction})</text>'
    )

    # Points.
    for row in rows:
        cls = "kept" if row.accepted else "discarded"
        r = 5.0 if row.accepted else 3.2
        parts.append(
            f'<circle class="{cls}" cx="{x_pos(row.timestamp.timestamp()):.1f}" '
            f'cy="{y_pos(row.metric):.1f}" r="{r:.1f}">'
            f'<title>cycle {row.cycle}: {escape(row.description)}; {metric}={row.metric:g}</title>'
            "</circle>"
        )

    # Running best step line from accepted rows.
    if improvements:
        coords: list[tuple[float, float]] = []
        for idx, row in enumerate(improvements):
            x = x_pos(row.timestamp.timestamp())
            y = y_pos(row.metric)
            if idx == 0:
                coords.append((x, y))
            else:
                coords.append((x, coords[-1][1]))
                coords.append((x, y))
        coords.append((x_pos(max_x), coords[-1][1]))
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(f'<polyline class="best" points="{point_text}"/>')

    # Final fitness callout geometry.
    final_x = x_pos(max_x)
    final_y = y_pos(best_row.metric)
    callout_w = 300
    callout_h = 72
    callout_x = min(width - margin_right - callout_w, max(margin_left + 20, final_x - callout_w - 16))
    callout_y = max(margin_top + 12, min(height - margin_bottom - callout_h - 12, final_y - callout_h - 12))

    # Labels for selected accepted improvements.
    selected_labels = select_improvement_labels(
        improvements,
        lower_is_better=lower_is_better,
        major_min_gain=major_min_gain,
        large_gain_fraction=large_gain_fraction,
        huge_gain_fraction=huge_gain_fraction,
        quiet_cycles=quiet_cycles,
        quiet_hours=quiet_hours,
        min_label_cycles=min_label_cycles,
        min_label_hours=min_label_hours,
        max_labels=max_labels,
    )
    occupied: list[tuple[float, float, float, float]] = [
        (callout_x - 8, callout_y - 8, callout_x + callout_w + 8, callout_y + callout_h + 8),
    ]

    def overlaps(box: tuple[float, float, float, float]) -> bool:
        left, top, right, bottom = box
        for other_left, other_top, other_right, other_bottom in occupied:
            separated = (
                right < other_left
                or left > other_right
                or bottom < other_top
                or top > other_bottom
            )
            if not separated:
                return True
        return False

    def place_label(
        point_x: float,
        point_y: float,
        text: str,
    ) -> tuple[float, float, str, tuple[float, float, float, float]]:
        text_w = max(48.0, min(320.0, len(text) * 7.5))
        text_h = 18.0
        anchor = "start"
        dx = 12.0
        if point_x + dx + text_w > width - margin_right:
            anchor = "end"
            dx = -16.0
        y_offsets = [-14.0, -38.0, 20.0, -62.0, 44.0, -86.0, 68.0]
        fallback: tuple[float, float, str, tuple[float, float, float, float]] | None = None
        for y_offset in y_offsets:
            label_x = point_x + dx
            label_y = point_y + y_offset
            label_y = max(margin_top + 24.0, min(height - margin_bottom - 8.0, label_y))
            if anchor == "start":
                box = (label_x - 4, label_y - text_h, label_x + text_w + 4, label_y + 4)
            else:
                box = (label_x - text_w - 4, label_y - text_h, label_x + 4, label_y + 4)
            candidate = (label_x, label_y, anchor, box)
            if fallback is None:
                fallback = candidate
            if not overlaps(box):
                return candidate
        assert fallback is not None
        return fallback

    for idx, label in enumerate(selected_labels):
        row = label.row
        point_x = x_pos(row.timestamp.timestamp())
        point_y = y_pos(row.metric)
        x, y, anchor, box = place_label(point_x, point_y, label.text)
        occupied.append(box)
        parts.append(
            f'<line class="label-line" x1="{point_x:.1f}" y1="{point_y:.1f}" '
            f'x2="{x:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="label" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">'
            f"{escape(label.text)}</text>"
        )

    # Final fitness callout.
    parts.append(
        f'<rect x="{callout_x:.1f}" y="{callout_y:.1f}" width="{callout_w}" height="{callout_h}" '
        f'rx="6" fill="white" stroke="#8fd6ad"/>'
    )
    parts.append(
        f'<text class="note-strong" x="{callout_x + 14:.1f}" y="{callout_y + 28:.1f}">'
        f"Final best: {best_row.metric:.6f}</text>"
    )
    parts.append(
        f'<text class="note" x="{callout_x + 14:.1f}" y="{callout_y + 52:.1f}">'
        f"cycle {best_row.cycle}, commit {escape(best_row.commit[:8])}</text>"
    )

    # Legend.
    legend_x = width - margin_right - 210
    legend_y = margin_top + 10
    parts.append(f'<rect x="{legend_x}" y="{legend_y - 22}" width="200" height="92" fill="white" stroke="#d5d5d5"/>')
    parts.append(f'<circle class="discarded" cx="{legend_x + 20}" cy="{legend_y}" r="4"/>')
    parts.append(f'<text class="legend" x="{legend_x + 40}" y="{legend_y + 5}">Discarded</text>')
    parts.append(f'<circle class="kept" cx="{legend_x + 20}" cy="{legend_y + 30}" r="5"/>')
    parts.append(f'<text class="legend" x="{legend_x + 40}" y="{legend_y + 35}">Kept</text>')
    parts.append(f'<line class="best" x1="{legend_x + 8}" y1="{legend_y + 60}" x2="{legend_x + 32}" y2="{legend_y + 60}"/>')
    parts.append(f'<text class="legend" x="{legend_x + 40}" y="{legend_y + 65}">Running best</text>')

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def font(size: int, *, bold: bool = False):
    try:
        from PIL import ImageFont
    except ModuleNotFoundError as exc:
        raise SystemExit("PNG output requires Pillow: install the pillow package") from exc

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_png(
    rows: list[Result],
    metric: str,
    output: Path,
    *,
    title: str | None,
    lower_is_better: bool,
    max_labels: int,
    major_min_gain: float,
    large_gain_fraction: float,
    huge_gain_fraction: float,
    quiet_cycles: int,
    quiet_hours: float,
    min_label_cycles: int,
    min_label_hours: float,
    width: int,
    height: int,
    experiment_count: int | None = None,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise SystemExit("PNG output requires Pillow: install the pillow package") from exc

    margin_left = 120
    margin_right = 70
    margin_top = 82
    margin_bottom = 105
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    times = [row.timestamp.timestamp() for row in rows]
    values = [row.metric for row in rows]
    min_x, max_x = min(times), max(times)
    min_y, max_y = min(values), max(values)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        pad = abs(min_y) * 0.01 or 0.01
        min_y -= pad
        max_y += pad
    y_pad = (max_y - min_y) * 0.08
    min_y -= y_pad
    max_y += y_pad

    def x_pos(ts: float) -> float:
        return margin_left + (ts - min_x) / (max_x - min_x) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + (max_y - value) / (max_y - min_y) * plot_h

    improvements = best_steps(rows, lower_is_better)
    improvement_count = max(0, len(improvements) - 1)
    final_row = rows[-1]
    best_row = improvements[-1] if improvements else final_row
    total = experiment_count if experiment_count is not None else len(rows)
    heading = title or f"Autoresearch Progress: {total} Experiments, {improvement_count} Kept Improvements"
    direction = "lower is better" if lower_is_better else "higher is better"

    green = (38, 153, 88)
    line_green = (95, 199, 139)
    dark_green = (13, 107, 58)
    gray = (180, 180, 180)
    grid = (230, 230, 230)
    text = (31, 41, 55)
    note = (51, 65, 85)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(30)
    note_font = font(18)
    tick_font = font(15)
    axis_font = font(22, bold=True)
    label_font = font(17, bold=True)
    legend_font = font(18)

    def centered(value: str, y: float, font_obj, fill=text) -> None:
        bbox = draw.textbbox((0, 0), value, font=font_obj)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), value, font=font_obj, fill=fill)

    centered(heading, 24, title_font)
    centered(
        f"Best {metric}: {best_row.metric:.6f} at cycle {best_row.cycle}; "
        f"latest recorded: {final_row.metric:.6f} at cycle {final_row.cycle}",
        59,
        note_font,
        note,
    )

    tick_count = 6
    for idx in range(tick_count + 1):
        y_value = min_y + (max_y - min_y) * idx / tick_count
        y = y_pos(y_value)
        draw.line((margin_left, y, width - margin_right, y), fill=grid + (255,), width=1)
        label = nice_number(y_value)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((margin_left - 14 - (bbox[2] - bbox[0]), y - 8), label, font=tick_font, fill=text)

    for idx in range(7):
        ts = min_x + (max_x - min_x) * idx / 6
        x = x_pos(ts)
        label = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        draw.line((x, margin_top, x, height - margin_bottom), fill=grid + (255,), width=1)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, height - margin_bottom + 24), label, font=tick_font, fill=text)

    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill=(30, 30, 30, 255), width=2)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill=(30, 30, 30, 255), width=2)
    centered("Time", height - 48, axis_font)
    y_axis_label = f"{metric} ({direction})"
    axis_image = Image.new("RGBA", (360, 34), (255, 255, 255, 0))
    axis_draw = ImageDraw.Draw(axis_image)
    axis_draw.text((0, 0), y_axis_label, font=axis_font, fill=text)
    axis_image = axis_image.rotate(90, expand=True)
    image.paste(axis_image, (26, int((height - axis_image.height) / 2)), axis_image)

    for row in rows:
        x = x_pos(row.timestamp.timestamp())
        y = y_pos(row.metric)
        r = 6 if row.accepted else 4
        fill = (50, 199, 122, 255) if row.accepted else gray + (110,)
        outline = dark_green + (255,) if row.accepted else gray + (90,)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2 if row.accepted else 1)

    if improvements:
        coords: list[tuple[float, float]] = []
        for idx, row in enumerate(improvements):
            x = x_pos(row.timestamp.timestamp())
            y = y_pos(row.metric)
            if idx == 0:
                coords.append((x, y))
            else:
                coords.append((x, coords[-1][1]))
                coords.append((x, y))
        coords.append((x_pos(max_x), coords[-1][1]))
        draw.line(coords, fill=line_green + (255,), width=4, joint="curve")

    final_x = x_pos(max_x)
    final_y = y_pos(best_row.metric)
    callout_w = 330
    callout_h = 84
    callout_x = min(width - margin_right - callout_w, max(margin_left + 20, final_x - callout_w - 16))
    callout_y = max(margin_top + 18, min(height - margin_bottom - callout_h - 14, final_y - callout_h - 14))

    selected_labels = select_improvement_labels(
        improvements,
        lower_is_better=lower_is_better,
        major_min_gain=major_min_gain,
        large_gain_fraction=large_gain_fraction,
        huge_gain_fraction=huge_gain_fraction,
        quiet_cycles=quiet_cycles,
        quiet_hours=quiet_hours,
        min_label_cycles=min_label_cycles,
        min_label_hours=min_label_hours,
        max_labels=max_labels,
    )
    occupied: list[tuple[float, float, float, float]] = [
        (callout_x - 8, callout_y - 8, callout_x + callout_w + 8, callout_y + callout_h + 8),
    ]

    def overlaps(box: tuple[float, float, float, float]) -> bool:
        left, top, right, bottom = box
        for other_left, other_top, other_right, other_bottom in occupied:
            if not (right < other_left or left > other_right or bottom < other_top or top > other_bottom):
                return True
        return False

    def place_label(point_x: float, point_y: float, label_text: str):
        bbox = draw.textbbox((0, 0), label_text, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        anchor = "start"
        dx = 14.0
        if point_x + dx + text_w > width - margin_right:
            anchor = "end"
            dx = -18.0
        y_offsets = [-16.0, -44.0, 24.0, -72.0, 52.0, -100.0, 80.0, -128.0, 108.0]
        fallback = None
        for y_offset in y_offsets:
            label_x = point_x + dx
            label_y = max(margin_top + 24.0, min(height - margin_bottom - 12.0, point_y + y_offset))
            if anchor == "start":
                box = (label_x - 5, label_y - text_h - 4, label_x + text_w + 5, label_y + 5)
            else:
                box = (label_x - text_w - 5, label_y - text_h - 4, label_x + 5, label_y + 5)
            candidate = (label_x, label_y, anchor, box)
            if fallback is None:
                fallback = candidate
            if not overlaps(box):
                return candidate
        return fallback

    for label in selected_labels:
        row = label.row
        point_x = x_pos(row.timestamp.timestamp())
        point_y = y_pos(row.metric)
        placed = place_label(point_x, point_y, label.text)
        if placed is None:
            continue
        label_x, label_y, anchor, box = placed
        occupied.append(box)
        draw.line((point_x, point_y, label_x, label_y), fill=green + (145,), width=1)
        if anchor == "end":
            bbox = draw.textbbox((0, 0), label.text, font=label_font)
            text_x = label_x - (bbox[2] - bbox[0])
        else:
            text_x = label_x
        draw.text((text_x, label_y - 16), label.text, font=label_font, fill=green)

    draw.rounded_rectangle((callout_x, callout_y, callout_x + callout_w, callout_y + callout_h), radius=8, fill=(255, 255, 255, 238), outline=(143, 214, 173, 255), width=2)
    draw.text((callout_x + 16, callout_y + 17), f"Final best: {best_row.metric:.6f}", font=font(20, bold=True), fill=dark_green)
    draw.text((callout_x + 16, callout_y + 48), f"cycle {best_row.cycle}, commit {best_row.commit[:8]}", font=note_font, fill=note)

    legend_x = width - margin_right - 220
    legend_y = margin_top + 8
    draw.rectangle((legend_x, legend_y - 18, legend_x + 210, legend_y + 86), fill=(255, 255, 255, 235), outline=(213, 213, 213, 255))
    draw.ellipse((legend_x + 17, legend_y, legend_x + 25, legend_y + 8), fill=gray + (110,))
    draw.text((legend_x + 44, legend_y - 5), "Discarded", font=legend_font, fill=text)
    draw.ellipse((legend_x + 15, legend_y + 31, legend_x + 27, legend_y + 43), fill=(50, 199, 122, 255), outline=dark_green + (255,), width=2)
    draw.text((legend_x + 44, legend_y + 27), "Kept", font=legend_font, fill=text)
    draw.line((legend_x + 11, legend_y + 68, legend_x + 33, legend_y + 68), fill=line_green + (255,), width=4)
    draw.text((legend_x + 44, legend_y + 58), "Running best", font=legend_font, fill=text)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", default="results.tsv", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("reports/autoresearch_progress.svg"))
    parser.add_argument("--metric", default="mean_rel_steps")
    parser.add_argument("--title")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="include invalid/crashed rows; by default they are omitted because they often use sentinel metrics",
    )
    parser.add_argument("--max-labels", type=int, default=40)
    parser.add_argument(
        "--major-min-gain",
        type=float,
        default=0.0,
        help="always label accepted improvements with at least this absolute metric gain",
    )
    parser.add_argument(
        "--large-gain-fraction",
        type=float,
        default=0.30,
        help="label improvements whose gain is at least this fraction of the largest gain",
    )
    parser.add_argument(
        "--huge-gain-fraction",
        type=float,
        default=0.65,
        help="always label improvements whose gain is at least this fraction of the largest gain",
    )
    parser.add_argument(
        "--quiet-cycles",
        type=int,
        default=55,
        help="label the next improvement after this many cycles without a label",
    )
    parser.add_argument(
        "--quiet-hours",
        type=float,
        default=7.0,
        help="label the next improvement after this many hours without a label",
    )
    parser.add_argument(
        "--min-label-cycles",
        type=int,
        default=12,
        help="minimum cycle separation for ordinary large-gain labels",
    )
    parser.add_argument(
        "--min-label-hours",
        type=float,
        default=1.5,
        help="minimum time separation for ordinary large-gain labels",
    )
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    all_rows = load_results(args.results, args.metric)
    rows = all_rows if args.include_invalid else [row for row in all_rows if row.is_valid]
    if not rows:
        raise SystemExit(f"{args.results} has no valid rows for {args.metric}")
    render_kwargs = dict(
        title=args.title,
        lower_is_better=not args.higher_is_better,
        max_labels=args.max_labels,
        major_min_gain=args.major_min_gain,
        large_gain_fraction=args.large_gain_fraction,
        huge_gain_fraction=args.huge_gain_fraction,
        quiet_cycles=args.quiet_cycles,
        quiet_hours=args.quiet_hours,
        min_label_cycles=args.min_label_cycles,
        min_label_hours=args.min_label_hours,
        width=args.width,
        height=args.height,
        experiment_count=len(all_rows),
    )
    if args.output.suffix.lower() == ".png":
        render_png(rows, args.metric, args.output, **render_kwargs)
    else:
        render_svg(rows, args.metric, args.output, **render_kwargs)
    omitted = len(all_rows) - len(rows)
    suffix = f" ({omitted} invalid rows omitted)" if omitted else ""
    print(f"wrote {args.output}{suffix}")


if __name__ == "__main__":
    main()
