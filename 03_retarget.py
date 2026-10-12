# Etape 3 : adapter les demos humaines au robot ("retargeting").
# Les demos sont donnees dans le repere du marqueur de chaque video (et les deux videos
# n'ont pas le meme repere). On les met toutes dans le repere du robot :
#   - la position moyenne de depart (A) devient le point A de la simulation
#   - la position moyenne d'arrivee (B) devient le point B de la simulation
# On garde la forme exacte du trajet et les petites variations d'une demo a l'autre.
# On ralentit aussi le geste (l'humain est tres rapide : 0,8 s).

import os
import numpy as np

DOSSIER_DEMOS = "data/demos"
DOSSIER_SORTIE = "data/demos_robot"

# positions de A et B dans la simulation (metres, repere du robot)
A_SIM = np.array([0.45, -0.17])
B_SIM = np.array([0.45, 0.18])

RALENTI = 3.0  # le robot rejoue le geste 3 fois plus lentement


def angle(v):
    return np.arctan2(v[1], v[0])


def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    fichiers = sorted(f for f in os.listdir(DOSSIER_DEMOS) if f.endswith(".npy"))

    # on regroupe par video (le debut du nom : IMG_3320, IMG_3321...)
    videos = sorted(set(f.split("_demo")[0] for f in fichiers))

    for video in videos:
        demos = {f: np.load(os.path.join(DOSSIER_DEMOS, f)) for f in fichiers if f.startswith(video)}

        # positions de A et B dans le repere du marqueur de cette video
        departs_A = [d[0, 1:] for f, d in demos.items() if "AversB" in f]
        arrivees_B = [d[-1, 1:] for f, d in demos.items() if "AversB" in f]
        A_vid = np.median(departs_A, axis=0)
        B_vid = np.median(arrivees_B, axis=0)

        # rotation qui aligne la direction A->B de la video sur celle de la simulation
        theta = angle(B_SIM - A_SIM) - angle(B_vid - A_vid)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        print(video, ": A->B =", round(np.linalg.norm(B_vid - A_vid), 3), "m, rotation",
              round(np.degrees(theta)), "deg")

        for f, d in demos.items():
            xy = (d[:, 1:] - A_vid) @ R.T + A_SIM
            sortie = np.column_stack([d[:, 0] * RALENTI, xy])
            np.save(os.path.join(DOSSIER_SORTIE, f), sortie)


if __name__ == "__main__":
    main()
