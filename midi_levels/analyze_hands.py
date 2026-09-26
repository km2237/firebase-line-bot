import json
import sys

import numpy as np
import pretty_midi

SRC = sys.argv[1] if len(sys.argv) > 1 else "turkish.mid"
name = pretty_midi.note_number_to_name

pm = pretty_midi.PrettyMIDI(SRC)
notes = sorted(pm.instruments[0].notes, key=lambda n: (n.start, n.pitch))
N = len(notes)
S = np.array([n.start for n in notes])
E = np.array([n.end for n in notes])
P = np.array([n.pitch for n in notes])

# --- onset clusters (chords): chain onsets < 35 ms apart, total spread < 150 ms
clusters, cur = [], [0]
for i in range(1, N):
    if S[i] - S[cur[-1]] < 0.035 and S[i] - S[cur[0]] < 0.15:
        cur.append(i)
    else:
        clusters.append(cur)
        cur = [i]
clusters.append(cur)
T = len(clusters)
cl_time = np.array([S[c[0]] for c in clusters])

# --- time-varying split point by Viterbi: path of least note density through the piano roll
# split s means LH <= s-1 < s <= RH
splits = np.arange(50, 77)
K = len(splits)
WIN = 0.6  # seconds of context on each side
SIGMA = 1.2
emis = np.zeros((T, K))
lo = hi = 0
for t in range(T):
    while S[lo] < cl_time[t] - WIN:
        lo += 1
    while hi < N and S[hi] <= cl_time[t] + WIN:
        hi += 1
    local = P[lo:hi][:, None]
    emis[t] = np.exp(-((local - (splits[None, :] - 0.5)) ** 2) / (2 * SIGMA**2)).sum(0)
    ps = P[clusters[t]]
    for j, s in enumerate(splits):
        lh, rh = ps[ps < s], ps[ps >= s]
        if len(lh) and (lh.max() - lh.min() > 14 or len(lh) > 5):
            emis[t, j] += 20
        if len(rh) and (rh.max() - rh.min() > 14 or len(rh) > 5):
            emis[t, j] += 20

hist = np.bincount(P, minlength=128)
s0 = int(splits[np.argmin([hist[s - 1] + hist[s] for s in splits])])
prior = 0.05 * np.abs(splits - s0)
LAM = 0.6
trans = LAM * np.abs(splits[:, None] - splits[None, :])
D = emis[0] + prior
back = np.zeros((T, K), dtype=int)
for t in range(1, T):
    scale = 0.25 if cl_time[t] - cl_time[t - 1] > 1.0 else 1.0
    tot = D[:, None] + trans * scale
    back[t] = np.argmin(tot, axis=0)
    D = tot[back[t], np.arange(K)] + emis[t] + prior
path = np.zeros(T, dtype=int)
path[-1] = int(np.argmin(D))
for t in range(T - 1, 0, -1):
    path[t - 1] = back[t, path[t]]
raw_split = splits[path]
# remove brief excursions: running median over +-SMOOTH seconds
SMOOTH = 1.0
split_at_cluster = np.array([
    int(np.median(raw_split[(cl_time >= t - SMOOTH) & (cl_time <= t + SMOOTH)])) for t in cl_time
])

hand = np.zeros(N, dtype=int)  # 1 = RH, 0 = LH
note_split = np.zeros(N, dtype=int)
for t, c in enumerate(clusters):
    for i in c:
        note_split[i] = split_at_cluster[t]
        hand[i] = 1 if P[i] >= split_at_cluster[t] else 0

# melodic-continuity fix: a lone note inside a fast stepwise run keeps the run's hand
fixed_by_run = 0
for t in range(1, T - 1):
    c = clusters[t]
    if len(c) != 1:
        continue
    i = c[0]
    cand = []
    for nb, dt in ((clusters[t - 1], S[i] - cl_time[t - 1]), (clusters[t + 1], cl_time[t + 1] - S[i])):
        if dt < 0.14:
            near = [j for j in nb if abs(P[j] - P[i]) <= 4]
            if near:
                cand.append(hand[near[0]])
    if len(cand) == 2 and cand[0] == cand[1] and cand[0] != hand[i]:
        hand[i] = cand[0]
        fixed_by_run += 1

fixed_split = np.where(P >= s0, 1, 0)
disagree = int((fixed_split != hand).sum())
ambiguous = int((np.abs(P - (note_split - 0.5)) < 2).sum())

# --- key per window (Krumhansl-Kessler), smoothed into sections
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PCN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def best_key(pcs):
    best, lab = -2, None
    for tonic in range(12):
        for prof, mode in ((KK_MAJ, "major"), (KK_MIN, "minor")):
            r = np.corrcoef(pcs, np.roll(prof, tonic))[0, 1]
            if r > best:
                best, lab = r, f"{PCN[tonic]} {mode}"
    return lab


step, half = 1.0, 3.0
centers = np.arange(0, S.max() + step, step)
keys = []
for c in centers:
    m = (S >= c - half) & (S < c + half)
    pcs = np.bincount(P[m] % 12, minlength=12).astype(float)
    keys.append(best_key(pcs) if pcs.sum() >= 12 else None)
sm = []
for k in range(len(keys)):
    w = [x for x in keys[max(0, k - 3): k + 4] if x]
    sm.append(max(set(w), key=w.count) if w else None)
sections = []
for c, k in zip(centers, sm):
    if sections and sections[-1]["key"] == k:
        sections[-1]["end"] = float(c + step)
    else:
        sections.append({"key": k, "start": float(max(0.0, c - step / 2)), "end": float(c + step)})
merged = []
for sec in sections:
    if merged and sec["end"] - sec["start"] < 5:
        merged[-1]["end"] = sec["end"]
    elif merged and merged[-1]["key"] == sec["key"]:
        merged[-1]["end"] = sec["end"]
    else:
        merged.append(sec)
merged[-1]["end"] = float(E.max())


# --- per-section, per-hand texture
def texture(mask_notes):
    idx = np.where(mask_notes)[0]
    if len(idx) == 0:
        return None
    dur = S[idx].max() - S[idx].min() + 1e-9
    cl = []
    for c in clusters:
        m = [i for i in c if mask_notes[i]]
        if m:
            cl.append(m)
    sizes = np.array([len(m) for m in cl])
    octs = sum(1 for m in cl if len(m) >= 2 and any((P[a] - P[b]) % 12 == 0 and P[a] != P[b] for a in m for b in m))
    onsets = np.array([S[m[0]] for m in cl])
    ioi = np.diff(onsets)
    single = sizes[:-1] == 1
    run16 = int(((ioi > 0.07) & (ioi < 0.13) & single).sum())
    return {
        "notes": int(len(idx)),
        "notes_per_sec": round(len(idx) / dur, 1),
        "range": f"{name(int(P[idx].min()))}-{name(int(P[idx].max()))}",
        "single_%": round(100 * (sizes == 1).mean()),
        "2note_%": round(100 * (sizes == 2).mean()),
        "3plus_%": round(100 * (sizes >= 3).mean()),
        "octave_%": round(100 * octs / len(cl)),
        "16th_run_%": round(100 * run16 / max(1, len(cl) - 1)),
        "median_ioi_ms": int(np.median(ioi) * 1000) if len(ioi) else None,
    }


report = {
    "file": SRC,
    "notes": N,
    "clusters": T,
    "global_valley_split": f"LH <= {name(s0 - 1)} / RH >= {name(s0)}",
    "notes_changed_vs_fixed_split": disagree,
    "notes_near_moving_split(+-2)": ambiguous,
    "run_continuity_fixes": fixed_by_run,
    "moving_split_range": f"{name(int(split_at_cluster.min()))}..{name(int(split_at_cluster.max()))}",
    "sections": [],
}
for sec in merged:
    m = (S >= sec["start"]) & (S < sec["end"])
    report["sections"].append({
        "time": f"{sec['start']:.0f}-{sec['end']:.0f}s",
        "key": sec["key"],
        "RH": texture(m & (hand == 1)),
        "LH": texture(m & (hand == 0)),
    })
print(json.dumps(report, ensure_ascii=False, indent=1))

# outputs for plotting / MIDI export
np.savez("hands.npz", S=S, E=E, P=P, hand=hand, note_split=note_split,
         cl_time=cl_time, split_at_cluster=split_at_cluster, s0=s0,
         sec_start=[s["start"] for s in merged], sec_end=[s["end"] for s in merged],
         sec_key=[s["key"] or "" for s in merged])
out = pretty_midi.PrettyMIDI(initial_tempo=120)
for h, label in ((1, "Right Hand"), (0, "Left Hand")):
    inst = pretty_midi.Instrument(program=0, name=label)
    inst.notes = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
                  for n, hh in zip(notes, hand) if hh == h]
    out.instruments.append(inst)
out.write(SRC.replace(".mid", "_hands_split.mid"))
