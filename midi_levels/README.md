# midi_levels

Turns an audio-transcribed solo piano MIDI into graded practice files (6 levels), so a hard
arrangement can be learned step by step.

```
pip install -r midi_levels/requirements.txt
python -m midi_levels song.mid                       # 2/4, quarter = 125-176 bpm
python -m midi_levels song.mid -o out --beats-per-bar 4 --min-bpm 60 --max-bpm 110
```

Pipeline:

1. **Hand separation** (`hands.py`) - a split pitch per chord found by Viterbi along the emptiest path
   through the piano roll, smoothed with a 1 s running median.
2. **Beat tracking / quantization** (`beats.py`) - 8th-note pulse by dynamic programming with a slowly
   varying period, split into 16ths; bar lines placed from left-hand bass notes per 2-bar block.
3. **Levels** (`levels.py`):

| Level | Right hand | Left hand |
|---|---|---|
| Lv1 | melody only (top voice, octaves folded) | - |
| Lv2 | melody only | lowest note on each beat |
| Lv3 | melody only | on each beat: bass + one note within a fifth |
| Lv4 | melody only | on each 8th: bass + one note within a fifth |
| Lv5 | melody only | as transcribed |
| Lv6 | as transcribed | as transcribed |

Output directory (default `<song>_levels/`): `<song>_hands.png` (piano roll coloured by hand),
`<song>_quantized.mid`, one `.mid` + `.musicxml` per level, and `report.json` with per-level density stats.

Limits: expects a single piano part with a steady 8th-note pulse. `--min-bpm/--max-bpm` must bracket the
real tempo; for slow pieces note values may come out halved (quarters written as 8ths).
Tuned on Mozart K.331 "Rondo alla Turca".
