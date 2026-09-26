import collections
import json
import sys

import mido

SRC = sys.argv[1]
src = mido.MidiFile(SRC)
TPB = src.ticks_per_beat
tempo = next(m.tempo for m in src.tracks[0] if m.type == "set_tempo")


def read_notes(track):
    t, active, notes = 0, {}, []
    for m in track:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            active[m.note] = (t, m.velocity)
        elif m.type in ("note_off", "note_on") and m.note in active:
            s, v = active.pop(m.note)
            notes.append([s, t, m.note, v])
    return sorted(notes)


RH = read_notes(src.tracks[1])
LH = read_notes(src.tracks[2])


def by_onset(notes):
    g = collections.defaultdict(list)
    for n in notes:
        g[n[0]].append(n)
    return g


def top_line(notes):
    """Single melody line: top note per onset, octaves folded to the lower note, register kept consistent."""
    g = by_onset(notes)
    ons = sorted(g)
    line = []
    for on in ons:
        pitches = {n[2] for n in g[on]}
        top = max(g[on], key=lambda x: x[2])
        p = top[2] - 12 if top[2] - 12 in pitches else top[2]
        line.append([on, top[1], p, top[3]])
    # broken octaves (octave played as two successive 16ths) -> keep the lower note
    merged = []
    for n in line:
        if merged and n[0] - merged[-1][0] <= TPB // 4 and abs(n[2] - merged[-1][2]) == 12:
            if n[2] < merged[-1][2]:
                merged[-1][2] = n[2]
            continue
        merged.append(n)
    # a note an octave away from both neighbours (dropped octave half in the transcription) -> move it back
    for k in range(1, len(merged) - 1):
        a, p, b = merged[k - 1][2], merged[k][2], merged[k + 1][2]
        for q in (p - 12, p + 12):
            if abs(p - a) >= 10 and abs(p - b) >= 10 and abs(q - a) <= 7 and abs(q - b) <= 7:
                merged[k][2] = q
    out = []
    for k, n in enumerate(merged):
        nxt = merged[k + 1][0] if k + 1 < len(merged) else n[0] + TPB
        out.append([n[0], max(min(n[1], nxt), n[0] + TPB // 4), n[2], n[3]])
    return out


def bass_on_beats(notes):
    """Lowest left-hand note on each beat, held until the next bass note (max one beat)."""
    g = by_onset(notes)
    beats = sorted(on for on in g if on % TPB == 0)
    out = []
    for k, on in enumerate(beats):
        n = min(g[on], key=lambda x: x[2])
        nxt = beats[k + 1] if k + 1 < len(beats) else on + TPB
        out.append([on, min(nxt, on + TPB), n[2], n[3]])
    return out


def lh_two_notes(notes, step):
    """Left hand only on multiples of `step` ticks, at most bass + one note within a fifth above it."""
    g = by_onset(notes)
    ons = sorted(on for on in g if on % step == 0)
    out = []
    for k, on in enumerate(ons):
        bass = min(g[on], key=lambda x: x[2])
        upper = [n for n in g[on] if 0 < n[2] - bass[2] <= 7]
        nxt = ons[k + 1] if k + 1 < len(ons) else on + step
        end = min(nxt, on + TPB)
        out.append([on, end, bass[2], bass[3]])
        if upper:
            top = max(upper, key=lambda x: x[2])
            out.append([on, end, top[2], top[3]])
    return out


LEVELS = {
    "Lv1_右手旋律のみ": (top_line(RH), []),
    "Lv2_右手旋律+左手低音": (top_line(RH), bass_on_beats(LH)),
    "Lv3_右手旋律+左手4分2音": (top_line(RH), lh_two_notes(LH, TPB)),
    "Lv4_右手旋律+左手8分2音": (top_line(RH), lh_two_notes(LH, TPB // 2)),
    "Lv5_右手旋律+左手原曲": (top_line(RH), LH),
    "Lv6_原曲": (RH, LH),
}


def write(path, rh, lh):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    mid.tracks.append(mido.MidiTrack([m.copy() for m in src.tracks[0]]))
    for name, notes, ch in (("Right Hand", rh, 0), ("Left Hand", lh, 1)):
        if not notes:
            continue
        ev = []
        for s, e, p, v in notes:
            ev.append((s, 1, mido.Message("note_on", note=p, velocity=v, channel=ch)))
            ev.append((e, 0, mido.Message("note_off", note=p, velocity=0, channel=ch)))
        ev.sort(key=lambda x: (x[0], x[1]))
        tr = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
        t = 0
        for tick, _, m in ev:
            tr.append(m.copy(time=tick - t))
            t = tick
        mid.tracks.append(tr)
    mid.save(path)


def stats(rh, lh):
    sec = max(n[1] for n in rh + lh) * tempo / 1e6 / TPB
    def hand(notes):
        if not notes:
            return None
        g = by_onset(notes)
        return {"音数": len(notes), "打鍵/秒": round(len(g) / sec, 1),
                "和音(2音以上)%": round(100 * sum(len(v) > 1 for v in g.values()) / len(g)),
                "最大同時音数": max(len(v) for v in g.values())}
    return {"右手": hand(rh), "左手": hand(lh)}


report = {}
for name, (rh, lh) in LEVELS.items():
    write(SRC.replace("_quantized.mid", f"_{name}.mid"), rh, lh)
    report[name] = stats(rh, lh)
print(json.dumps(report, ensure_ascii=False, indent=1))
