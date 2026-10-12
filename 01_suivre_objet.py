# Etape 1 : suivre l'objet (la barre verte) image par image.
# On cherche les pixels verts dans l'image, on prend le centre du plus gros morceau,
# puis on le convertit en metres sur la table avec l'homographie de l'etape 0.
# Resultat : un fichier .npy par video, de forme (nb_images, 3) = temps (s), x (m), y (m).
# Quand l'objet n'est pas vu (cache par la main), on met NaN.

import os
import cv2
import numpy as np

DOSSIER_VIDEOS = "data/videos"
DOSSIER_CALIB = "data/calibration"
DOSSIER_SUIVI = "data/suivi"

# plage de vert en HSV (teinte, saturation, valeur) -- a ajuster si besoin
VERT_MIN = np.array([35, 90, 60])
VERT_MAX = np.array([85, 255, 255])
SURFACE_MIN = 40  # en pixels, en dessous on considere que ce n'est pas l'objet


def position_objet(image):
    """Renvoie (x, y) en pixels du centre de la zone verte, ou None."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv, VERT_MIN, VERT_MAX)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    nb, etiquettes, stats, centres = cv2.connectedComponentsWithStats(masque)
    if nb <= 1:
        return None

    # on ignore le fond (indice 0) et on prend la plus grosse zone
    meilleur = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[meilleur, cv2.CC_STAT_AREA] < SURFACE_MIN:
        return None
    return centres[meilleur]


def main():
    os.makedirs(DOSSIER_SUIVI, exist_ok=True)

    for nom in sorted(os.listdir(DOSSIER_VIDEOS)):
        if not nom.lower().endswith((".mp4", ".mov")):
            continue
        base = nom.rsplit(".", 1)[0]

        chemin_H = os.path.join(DOSSIER_CALIB, base + "_H.npy")
        if not os.path.exists(chemin_H):
            print(nom, ": pas de calibration, lance d'abord 00_calibration.py")
            continue
        H = np.load(chemin_H)

        cap = cv2.VideoCapture(os.path.join(DOSSIER_VIDEOS, nom))
        fps = cap.get(cv2.CAP_PROP_FPS)
        resultat = []
        numero = 0

        while True:
            ok, image = cap.read()
            if not ok:
                break
            t = numero / fps
            p = position_objet(image)
            if p is None:
                resultat.append([t, np.nan, np.nan])
            else:
                # pixels -> metres sur la table
                m = cv2.perspectiveTransform(np.array([[p]], dtype=np.float32), H)[0][0]
                resultat.append([t, m[0], m[1]])
            numero += 1
        cap.release()

        resultat = np.array(resultat)
        np.save(os.path.join(DOSSIER_SUIVI, base + ".npy"), resultat)
        vu = np.sum(~np.isnan(resultat[:, 1]))
        print(nom, ":", len(resultat), "images, objet vu sur", vu)


if __name__ == "__main__":
    main()
