import collections

import mido


def by_onset(notes):
    g = collections.defaultdict(list)
    for n in notes:
        g[n[0]].append(n)
    return g


def melody(notes, tpb):
    """Single melody line: top note per onset, octaves folded to the lower note, register kept consistent."""
    g = by_onset(notes)
    line = []
    for on in sorted(g):
        pitches = {n[2] for n in g[on]}
        top = max(g[on], key=lambda x: x[2])
        p = top[2] - 12 if top[2] - 12 in pitches else top[2]
        line.append([on, top[1], p, top[3]])
    # broken octaves (an octave played as two successive 16ths) -> keep the lower note
    merged = []
    for n in line:
        if merged and n[0] - merged[-1][0] <= tpb // 4 and abs(n[2] - merged[-1][2]) == 12:
            merged[-1][2] = min(n[2], merged[-1][2])
            continue
        merged.append(n)
    # a note an octave away from both neighbours (octave half missing in the transcription) -> move it back
    for k in range(1, len(merged) - 1):
        a, p, b = merged[k - 1][2], merged[k][2], merged[k + 1][2]
        for q in (p - 12, p + 12):
            if abs(p - a) >= 10 and abs(p - b) >= 10 and abs(q - a) <= 7 and abs(q - b) <= 7:
                merged[k][2] = q
    out = []
    for k, n in enumerate(merged):
        nxt = merged[k + 1][0] if k + 1 < len(merged) else n[0] + tpb
        out.append([n[0], max(min(n[1], nxt), n[0] + tpb // 4), n[2], n[3]])
    return out


def bass_on_beats(notes, tpb):
    """Lowest note on each beat, held until the next one (at most one beat)."""
    g = by_onset(notes)
    beats = sorted(on for on in g if on % tpb == 0)
    out = []
    for k, on in enumerate(beats):
        n = min(g[on], key=lambda x: x[2])
        nxt = beats[k + 1] if k + 1 < len(beats) else on + tpb
        out.append([on, min(nxt, on + tpb), n[2], n[3]])
    return out


def two_notes(notes, step, tpb):
    """Only onsets on multiples of `step` ticks, at most the bass plus one note within a fifth above it."""
    g = by_onset(notes)
    ons = sorted(on for on in g if on % step == 0)
    out = []
    for k, on in enumerate(ons):
        bass = min(g[on], key=lambda x: x[2])
        upper = [n for n in g[on] if 0 < n[2] - bass[2] <= 7]
        nxt = ons[k + 1] if k + 1 < len(ons) else on + step
        end = min(nxt, on + tpb)
        out.append([on, end, bass[2], bass[3]])
        if upper:
            top = max(upper, key=lambda x: x[2])
            out.append([on, end, top[2], top[3]])
    return out


def make_levels(rh, lh, tpb):
    mel = melody(rh, tpb)
    return {
        "Lv1_右手旋律のみ": (mel, []),
        "Lv2_右手旋律+左手低音": (mel, bass_on_beats(lh, tpb)),
        "Lv3_右手旋律+左手4分2音": (mel, two_notes(lh, tpb, tpb)),
        "Lv4_右手旋律+左手8分2音": (mel, two_notes(lh, tpb // 2, tpb)),
        "Lv5_右手旋律+左手原曲": (mel, lh),
        "Lv6_原曲": (rh, lh),
    }


def write_midi(path, rh, lh, tpb, bpm, beats_per_bar):
    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=beats_per_bar, denominator=4, time=0))
    mid.tracks.append(meta)
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


def stats(rh, lh, tpb, bpm):
    end = max((n[1] for n in rh + lh), default=0)
    sec = end / tpb * 60 / bpm or 1

    def hand(notes):
        if not notes:
            return None
        g = by_onset(notes)
        return {"notes": len(notes), "strikes_per_sec": round(len(g) / sec, 1),
                "chord_pct": round(100 * sum(len(v) > 1 for v in g.values()) / len(g)),
                "max_simultaneous": max(len(v) for v in g.values())}
    return {"right": hand(rh), "left": hand(lh)}
