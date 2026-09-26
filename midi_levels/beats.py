import numpy as np

FPS = 100


def drop_isolated(S, gap=2.0):
    """Mask of notes that have another note within `gap` seconds (stray transcription noise is False)."""
    return np.array([np.sum(np.abs(S - s) < gap) > 1 for s in S])


def track_eighths(S, E, min_bpm=125, max_bpm=176):
    """8th-note pulse by dynamic programming over onset frames with a slowly varying period."""
    env = np.zeros(int(E.max() * FPS) + 60)
    for s in S:
        env[int(round(s * FPS))] += 1.0
    env = np.convolve(np.sqrt(env), [0.3, 0.7, 1.0, 0.7, 0.3], mode="same")
    taus = np.arange(int(FPS * 30 / max_bpm), int(FPS * 30 / min_bpm) + 1)
    F, J = len(env), len(taus)
    score = np.full((F, J), -1e9)
    back = np.full((F, J, 2), -1, dtype=int)
    first_f = int(S.min() * FPS)
    for f in range(F):
        for j, tau in enumerate(taus):
            best, arg = (0.0 if f <= first_f + 30 else -1e9), (-1, -1)
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
    lo = max(0, last_f - 60)
    f, j = np.unravel_index(np.argmax(score[lo:last_f + 10]), score[lo:last_f + 10].shape)
    f += lo
    eighths = []
    while f >= 0:
        eighths.append(f / FPS)
        f, j = back[f, j]
    eighths = np.array(eighths[::-1])
    t8 = float(np.median(np.diff(eighths)))
    return np.concatenate([eighths[0] - t8 * np.arange(4, 0, -1), eighths, eighths[-1] + t8 * np.arange(1, 5)])


def quantize(S, E, P, V, H, tpb=480, beats_per_bar=2, min_bpm=125, max_bpm=176):
    """Snap notes to a 16th grid built from the 8th pulse and place bar lines from left-hand bass notes.

    Returns (notes, bpm, info) where notes = {1: [[on, off, pitch, vel], ...], 0: [...]} in ticks.
    """
    six = tpb // 4
    bar16 = beats_per_bar * 4
    eighths = track_eighths(S, E, min_bpm, max_bpm)
    grid = np.concatenate([np.linspace(a, b, 2, endpoint=False) for a, b in zip(eighths[:-1], eighths[1:])]
                          + [[eighths[-1]]])
    on_idx = np.abs(S[:, None] - grid[None, :]).argmin(1)
    off_idx = np.maximum(np.abs(E[:, None] - grid[None, :]).argmin(1), on_idx + 1)
    resid_ms = (S - grid[on_idx]) * 1000

    # which 8th of the bar is the downbeat, decided per 2-bar block with a switching penalty
    lh = np.where(H == 0)[0]
    bass = [i for i in lh if P[i] <= P[lh[np.abs(S[lh] - S[i]) < 0.4]].min()]
    blk = bar16 * 2
    nblk = int(on_idx.max() // blk) + 1
    ev = np.zeros((nblk, bar16))
    for i in bass:
        ev[on_idx[i] // blk, on_idx[i] % bar16] += 1
    cand = list(range(0, bar16, 2))
    C = len(cand)

    def phase_score(b, k):
        s = 2 * ev[b, k]
        for j in range(1, beats_per_bar):
            s += ev[b, (k + 4 * j) % bar16]
        for j in range(beats_per_bar):
            s -= ev[b, (k + 2 + 4 * j) % bar16]
        return s

    blk_score = np.array([[phase_score(b, k) for k in cand] for b in range(nblk)])
    acc = blk_score[0].copy()
    bk = np.zeros((nblk, C), dtype=int)
    for b in range(1, nblk):
        tot = acc[:, None] - 4.0 * (1 - np.eye(C))
        bk[b] = tot.argmax(0)
        acc = tot.max(0) + blk_score[b]
    ph = np.zeros(nblk, dtype=int)
    ph[-1] = int(acc.argmax())
    for b in range(nblk - 1, 0, -1):
        ph[b - 1] = bk[b, ph[b]]
    blk_phase = np.array(cand)[ph]

    bar_ticks = tpb * beats_per_bar
    g_tick = np.zeros(len(grid), dtype=int)
    first = int(on_idx.min())
    tick = bar_ticks - six * ((cand[ph[0]] - first) % bar16)
    cur = blk_phase[0]
    phase_changes = []
    for g in range(len(grid)):
        b = min(g // blk, nblk - 1)
        if blk_phase[b] != cur:
            shift = (blk_phase[b] - cur) % bar16
            tick -= shift * six if shift <= bar16 // 2 else -(bar16 - shift) * six
            phase_changes.append(round(float(grid[g]), 1))
            cur = blk_phase[b]
        g_tick[g] = tick + (g - first) * six

    notes = {}
    for h in (1, 0):
        evs = {}
        for i in np.where(H == h)[0]:
            on = int(g_tick[on_idx[i]])
            off = max(int(g_tick[min(off_idx[i], len(grid) - 1)]), on + six)
            key = (on, int(P[i]))
            evs[key] = max(evs.get(key, (0, 0)), (off, int(V[i])))
        notes[h] = sorted([on, off, p, v] for (on, p), (off, v) in evs.items())

    bpm = 60.0 / (2 * float(np.median(np.diff(eighths))))
    info = {
        "tempo_bpm": round(bpm, 1),
        "bars": int(g_tick.max() // bar_ticks),
        "bar_phase_changes_at_sec": phase_changes,
        "onset_error_ms": {"median": round(float(np.median(np.abs(resid_ms))), 1),
                           "p90": round(float(np.percentile(np.abs(resid_ms), 90)), 1)},
    }
    return notes, bpm, info
