import json
import sys

import mido
import numpy as np
import pretty_midi

FPS = 100
TPB = 480
SIXTEENTH = TPB // 4
BAR = TPB * 2  # 2/4

d = np.load("hands.npz")
S, E, P, H = d["S"], d["E"], d["P"], d["hand"]
V = np.array([n.velocity for n in sorted(pretty_midi.PrettyMIDI(sys.argv[1]).instruments[0].notes,
                                           key=lambda n: (n.start, n.pitch))])

# drop isolated noise notes (nothing else within 2 s)
keep = np.array([np.sum(np.abs(S - s) < 2.0) > 1 for s in S])
S, E, P, H, V = S[keep], E[keep], P[keep], H[keep], V[keep]

# 8th-note grid by dynamic programming with a slowly varying period, then split each 8th into two 16ths
FPS = 100
env = np.zeros(int(E.max() * FPS) + 60)
for s in S:
    env[int(round(s * FPS))] += 1.0
env = np.convolve(np.sqrt(env), [0.3, 0.7, 1.0, 0.7, 0.3], mode="same")
taus = np.arange(17, 25)  # 8th-note period in frames (quarter = 125..176 bpm)
F, J = len(env), len(taus)
score = np.full((F, J), -1e9)
back = np.full((F, J, 2), -1, dtype=int)
first_f = int(S.min() * FPS)
for f in range(F):
    for j, tau in enumerate(taus):
        if f <= first_f + 30:
            best, arg = 0.0, (-1, -1)
        else:
            best, arg = -1e9, (-1, -1)
        p = f - tau
        if p >= 0:
            for jj in (j - 1, j, j + 1):
                if 0 <= jj < J:
                    v = score[p, jj] - (0.3 if jj != j else 0.0)
                    if v > best:
                        best, arg = v, (p, jj)
        if best > -1e8:
            score[f, j] = env[f] + best
            back[f, j] = arg
last_f = int(E.max() * FPS)
f, j = np.unravel_index(np.argmax(score[max(0, last_f - 60):last_f + 10]), score[max(0, last_f - 60):last_f + 10].shape)
f += max(0, last_f - 60)
eighths = []
while f >= 0:
    eighths.append(f / FPS)
    f, j = back[f, j]
eighths = np.array(eighths[::-1])
T8 = float(np.median(np.diff(eighths)))
eighths = np.concatenate([eighths[0] - T8 * np.arange(4, 0, -1), eighths, eighths[-1] + T8 * np.arange(1, 5)])
grid = np.concatenate([np.linspace(a, b, 2, endpoint=False) for a, b in zip(eighths[:-1], eighths[1:])] + [[eighths[-1]]])
beats = grid[::4]
ibi = float(np.median(np.diff(beats)))
on_idx = np.abs(S[:, None] - grid[None, :]).argmin(1)
resid_ms = (S - grid[on_idx]) * 1000
off_idx = np.maximum(np.abs(E[:, None] - grid[None, :]).argmin(1), on_idx + 1)

# bar phase (which 8th is the downbeat), decided per 2-bar block with a switching penalty
lh = np.where(H == 0)[0]
bass = [i for i in lh if P[i] <= P[lh[np.abs(S[lh] - S[i]) < 0.4]].min()]
BLK = 16
nblk = int(on_idx.max() // BLK) + 1
ev = np.zeros((nblk, 8))
for i in bass:
    ev[on_idx[i] // BLK, on_idx[i] % 8] += 1
cand = [0, 2, 4, 6]
blk_score = np.array([[2 * ev[b, k] + ev[b, (k + 4) % 8] - ev[b, (k + 2) % 8] - ev[b, (k + 6) % 8] for k in cand]
                      for b in range(nblk)])
SW = 4.0
acc = blk_score[0].copy()
bk = np.zeros((nblk, 4), dtype=int)
for b in range(1, nblk):
    tot = acc[:, None] - SW * (1 - np.eye(4))
    bk[b] = tot.argmax(0)
    acc = tot.max(0) + blk_score[b]
ph = np.zeros(nblk, dtype=int)
ph[-1] = int(acc.argmax())
for b in range(nblk - 1, 0, -1):
    ph[b - 1] = bk[b, ph[b]]
blk_phase = np.array(cand)[ph]
# map grid index -> tick; each phase change shifts later music so its downbeats land on bar lines
g_tick = np.zeros(len(grid) + 1, dtype=int)
phase_changes = []
tick = BAR - SIXTEENTH * ((cand[ph[0]] - on_idx.min()) % 8)
cur = blk_phase[0]
for g in range(len(grid)):
    b = min(g // BLK, nblk - 1)
    if blk_phase[b] != cur:
        shift = (blk_phase[b] - cur) % 8
        tick -= shift * SIXTEENTH if shift <= 4 else -(8 - shift) * SIXTEENTH
        phase_changes.append(round(float(grid[g]), 1))
        cur = blk_phase[b]
    g_tick[g] = tick + (g - on_idx.min()) * SIXTEENTH
offset_grid = 0
bpm = 60.0 / ibi
mid = mido.MidiFile(ticks_per_beat=TPB)
meta = mido.MidiTrack()
meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
meta.append(mido.MetaMessage("time_signature", numerator=2, denominator=4, time=0))
mid.tracks.append(meta)
for h, label in ((1, "Right Hand"), (0, "Left Hand")):
    evs = {}
    for i in np.where(H == h)[0]:
        on = g_tick[on_idx[i]]
        off = max(g_tick[min(off_idx[i], len(grid)-1)], on + SIXTEENTH)
        key = (on, int(P[i]))
        evs[key] = max(evs.get(key, (0, 0)), (off, int(V[i])))
    msgs = []
    for (on, p), (off, v) in evs.items():
        msgs.append((on, 1, mido.Message("note_on", note=p, velocity=v, channel=0 if h else 1)))
        msgs.append((off, 0, mido.Message("note_off", note=p, velocity=0, channel=0 if h else 1)))
    msgs.sort(key=lambda x: (x[0], x[1]))
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=label, time=0))
    t = 0
    for tick, _, m in msgs:
        tr.append(m.copy(time=tick - t))
        t = tick
    mid.tracks.append(tr)
mid.save(sys.argv[1].replace(".mid", "_quantized.mid"))

ioi = np.diff(beats)
local_bpm = 60 / ioi
print(json.dumps({
    "removed_noise_notes": int((~keep).sum()),
    "tempo_bpm_median": round(bpm, 1),
    "beats": len(beats),
    "bars": int(g_tick.max() // BAR), "phase_changes_at_sec": phase_changes,
    "onset_residual_ms_abs": {"median": round(float(np.median(np.abs(resid_ms))), 1),
                               "p90": round(float(np.percentile(np.abs(resid_ms), 90)), 1),
                               "over_30ms_%": round(float((np.abs(resid_ms) > 30).mean() * 100), 1)},
    "bpm_range_p5_p95": [round(float(np.percentile(local_bpm, 5)), 1), round(float(np.percentile(local_bpm, 95)), 1)],
}, ensure_ascii=False, indent=1))
np.savez("beats.npz", beats=beats, resid_ms=resid_ms, S=S, P=P, H=H, on_tick=g_tick[on_idx])
