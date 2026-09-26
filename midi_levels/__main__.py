import argparse
import json
import sys
from pathlib import Path

from .beats import drop_isolated, quantize
from .hands import load_notes, separate_hands
from .levels import make_levels, stats, write_midi

TPB = 480


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m midi_levels",
        description="Turn an audio-transcribed solo piano MIDI into graded practice MIDI files (6 levels).")
    ap.add_argument("midi", help="input MIDI (single piano part, e.g. from an audio-to-MIDI transcriber)")
    ap.add_argument("-o", "--out", help="output directory (default: <input name>_levels next to the input)")
    ap.add_argument("--beats-per-bar", type=int, default=2, help="quarter-note beats per bar (default 2 = 2/4)")
    ap.add_argument("--min-bpm", type=float, default=125, help="slowest expected quarter-note tempo (default 125)")
    ap.add_argument("--max-bpm", type=float, default=176, help="fastest expected quarter-note tempo (default 176)")
    ap.add_argument("--no-musicxml", action="store_true", help="skip MusicXML export (needs music21)")
    ap.add_argument("--no-plot", action="store_true", help="skip the piano-roll image (needs matplotlib)")
    args = ap.parse_args(argv)

    src = Path(args.midi)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_levels")
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    S, E, P, V = load_notes(str(src))
    keep = drop_isolated(S)
    S, E, P, V = S[keep], E[keep], P[keep], V[keep]
    print(f"[1/4] {len(S)} notes loaded ({int((~keep).sum())} isolated noise notes removed)", file=sys.stderr)

    H, cl_time, split = separate_hands(S, P)
    print(f"[2/4] hands separated: right {int((H == 1).sum())}, left {int((H == 0).sum())}", file=sys.stderr)
    if not args.no_plot:
        from .export import plot_hands
        plot_hands(str(out / f"{stem}_hands.png"), stem, S, E, P, H, cl_time, split)

    notes, bpm, info = quantize(S, E, P, V, H, tpb=TPB, beats_per_bar=args.beats_per_bar,
                                  min_bpm=args.min_bpm, max_bpm=args.max_bpm)
    write_midi(str(out / f"{stem}_quantized.mid"), notes[1], notes[0], TPB, bpm, args.beats_per_bar)
    print(f"[3/4] quantized: {info['bars']} bars, tempo {info['tempo_bpm']} bpm, "
          f"median onset error {info['onset_error_ms']['median']} ms", file=sys.stderr)

    report = {"input": str(src), "quantize": info, "levels": {}}
    for name, (rh, lh) in make_levels(notes[1], notes[0], TPB).items():
        mid_path = out / f"{stem}_{name}.mid"
        write_midi(str(mid_path), rh, lh, TPB, bpm, args.beats_per_bar)
        if not args.no_musicxml:
            from .export import to_musicxml
            to_musicxml(str(mid_path), str(mid_path.with_suffix(".musicxml")))
        report["levels"][name] = stats(rh, lh, TPB, bpm)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[4/4] {len(report['levels'])} levels written to {out}", file=sys.stderr)
    print(json.dumps(report["levels"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
