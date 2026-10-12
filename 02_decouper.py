# Etape 2 : decouper la longue video en repetitions.
# On regarde la vitesse de l'objet : quand il bouge, c'est un transport.
# Chaque transport (de la position de depart a la position d'arrivee) devient une demo.
# Les noms de fichiers finissent par AversB ou BversA selon le sens.
# Resultat : un fichier .npy par demo dans data/demos/, de forme (n, 3) = temps (s), x (m), y (m).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOSSIER_SUIVI = "data/suivi"
DOSSIER_DEMOS = "data/demos"
DOSSIER_RESULTATS = "resultats"

SEUIL_BORD = 0.25        # s < 0.25 : en A ; s > 0.75 : en B
DUREE_MIN = 0.3          # s
MARGE = 0.3              # s : on garde un peu avant/apres le mouvement


def lisser(v, n=7):
    """Moyenne glissante simple. On remplit d'abord les trous (NaN)."""
    v = v.copy()
    trous = np.isnan(v)
    if trous.all():
        return v
    v[trous] = np.interp(np.flatnonzero(trous), np.flatnonzero(~trous), v[~trous])
    noyau = np.ones(n) / n
    # on ajoute des bords pour ne pas avoir d'effet de bord
    v = np.concatenate([np.full(n, v[0]), v, np.full(n, v[-1])])
    return np.convolve(v, noyau, mode="same")[n:-n]


def decouper(donnees):
    """L'objet fait des allers-retours entre deux endroits (A et B).
    On projette sa position sur la droite A-B : on obtient un nombre s entre 0 et 1.
    s proche de 0 = objet en A, s proche de 1 = objet en B.
    Un transport = le moment ou on quitte A et ou on arrive en B (ou l'inverse)."""
    t = donnees[:, 0]
    x = lisser(donnees[:, 1])
    y = lisser(donnees[:, 2])
    dt = t[1] - t[0]

    # les deux endroits : on prend les positions extremes (percentiles pour eviter le bruit)
    xy = np.stack([x, y], axis=1)
    direction = xy[:, 0] - xy[:, 0].min()  # premiere estimation de l'axe
    A = np.median(xy[direction < np.percentile(direction, 20)], axis=0)
    B = np.median(xy[direction > np.percentile(direction, 80)], axis=0)
    axe = B - A
    s = ((xy - A) @ axe) / (axe @ axe)

    # etat : 0 = en A, 1 = en B, -1 = entre les deux
    etat = np.full(len(s), -1)
    etat[s < SEUIL_BORD] = 0
    etat[s > 1 - SEUIL_BORD] = 1

    # on parcourt les images : on retient le dernier etat "stable" connu
    demos = []
    dernier_etat = None
    derniere_image = 0
    marge = int(MARGE / dt)
    for i in range(len(etat)):
        if etat[i] == -1:
            continue
        if dernier_etat is not None and etat[i] != dernier_etat:
            a = max(0, derniere_image - marge)
            b = min(len(t) - 1, i + marge)
            if (b - a) * dt >= DUREE_MIN:
                demos.append((np.stack([t[a:b] - t[a], x[a:b], y[a:b]], axis=1),
                              "AversB" if dernier_etat == 0 else "BversA"))
        dernier_etat = etat[i]
        derniere_image = i
    return demos, (t, x, y, s, etat)


def main():
    os.makedirs(DOSSIER_DEMOS, exist_ok=True)
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

    for nom in sorted(os.listdir(DOSSIER_SUIVI)):
        base = nom.rsplit(".", 1)[0]
        donnees = np.load(os.path.join(DOSSIER_SUIVI, nom))
        demos, (t, x, y, s, etat) = decouper(donnees)

        for i, (d, sens) in enumerate(demos):
            np.save(os.path.join(DOSSIER_DEMOS, f"{base}_demo{i:02d}_{sens}.npy"), d)
        print(base, ":", len(demos), "demos, duree moyenne",
              round(np.mean([d[-1, 0] for d, _ in demos]), 2), "s")

        # graphique de controle pour verifier les coupes a l'oeil
        plt.figure(figsize=(12, 4))
        plt.plot(t, x, label="x")
        plt.plot(t, y, label="y")
        plt.plot(t, s * 0.3, "k", lw=0.5, label="position le long de A-B (0 a 1, x0.3)")
        plt.xlabel("temps (s)")
        plt.ylabel("position (m)")
        plt.title(base + " : " + str(len(demos)) + " demos")
        plt.legend()
        plt.savefig(os.path.join(DOSSIER_RESULTATS, base + "_decoupage.png"), dpi=100)
        plt.close()


if __name__ == "__main__":
    main()
