import numpy as np
import pretty_midi


def load_notes(path):
    """All notes of the file as arrays sorted by (onset, pitch): start/end seconds, pitch, velocity."""
    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for inst in pm.instruments if not inst.is_drum for n in inst.notes),
                   key=lambda n: (n.start, n.pitch))
    return (np.array([n.start for n in notes]), np.array([n.end for n in notes]),
            np.array([n.pitch for n in notes]), np.array([n.velocity for n in notes]))


def onset_clusters(S, chain=0.035, spread=0.15):
    clusters, cur = [], [0]
    for i in range(1, len(S)):
        if S[i] - S[cur[-1]] < chain and S[i] - S[cur[0]] < spread:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


def separate_hands(S, P, win=0.6, sigma=1.2, lam=0.6, smooth=1.0):
    """Assign each note to a hand (1 = right, 0 = left).

    A split pitch per chord is found by Viterbi: it follows the emptiest path through the piano roll
    (few notes near the split, no hand spanning more than 14 semitones or 5 notes), then a running
    median over +-`smooth` seconds removes brief excursions.
    """
    N = len(S)
    clusters = onset_clusters(S)
    T = len(clusters)
    cl_time = np.array([S[c[0]] for c in clusters])
    splits = np.arange(50, 77)
    K = len(splits)

    emis = np.zeros((T, K))
    lo = hi = 0
    for t in range(T):
        while S[lo] < cl_time[t] - win:
            lo += 1
        while hi < N and S[hi] <= cl_time[t] + win:
            hi += 1
        local = P[lo:hi][:, None]
        emis[t] = np.exp(-((local - (splits[None, :] - 0.5)) ** 2) / (2 * sigma**2)).sum(0)
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
    trans = lam * np.abs(splits[:, None] - splits[None, :])
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
    raw = splits[path]
    split = np.array([int(np.median(raw[(cl_time >= t - smooth) & (cl_time <= t + smooth)])) for t in cl_time])

    hand = np.zeros(N, dtype=int)
    for t, c in enumerate(clusters):
        for i in c:
            hand[i] = int(P[i] >= split[t])
    return hand, cl_time, split
