# midi_levels

Generates graded practice MIDI files (4 levels) from an audio-transcribed solo piano MIDI.

```
pip install -r requirements.txt
python analyze_hands.py song.mid        # left/right hand separation -> hands.npz, song_hands_split.mid
python quantize.py song.mid             # 8th-note beat tracking, 2/4 bars -> song_quantized.mid
python make_levels.py song_quantized.mid
```

| Level | Right hand | Left hand |
|---|---|---|
| Lv1 | melody only (octaves folded, top voice) | - |
| Lv2 | melody only | lowest note on each beat |
| Lv3 | melody only | as transcribed |
| Lv4 | as transcribed | as transcribed |

Assumptions: single-track piano MIDI, 2/4 meter, steady 8th-note pulse (tuned on Mozart K.331 "Rondo alla Turca").
