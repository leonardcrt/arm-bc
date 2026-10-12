# Etape 0 : calibration avec le marqueur ArUco.
# Le marqueur fait 10 cm de cote et il est pose a plat sur la table.
# On detecte ses 4 coins dans l'image, et on en deduit une "homographie" :
# c'est une transformation qui passe d'un point de l'image (en pixels)
# a un point de la table (en metres), avec l'origine au coin du marqueur.

import os
import cv2
import numpy as np

DOSSIER_VIDEOS = "data/videos"
DOSSIER_CALIB = "data/calibration"
COTE_MARQUEUR = 0.10  # metres (100 mm)
ID_MARQUEUR = 0

# coins du marqueur dans le repere de la table (meme ordre qu'OpenCV :
# haut-gauche, haut-droite, bas-droite, bas-gauche)
COINS_TABLE = np.array([[0, 0], [COTE_MARQUEUR, 0],
                        [COTE_MARQUEUR, COTE_MARQUEUR], [0, COTE_MARQUEUR]],
                       dtype=np.float32)


def trouver_marqueur(chemin_video):
    """Cherche le marqueur dans les images de la video.
    On teste une image sur 15 et on garde la mediane des coins trouves."""
    dico = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detecteur = cv2.aruco.ArucoDetector(dico)

    cap = cv2.VideoCapture(chemin_video)
    coins_trouves = []
    dernier_image = None
    numero = 0

    while True:
        ok, image = cap.read()
        if not ok:
            break
        if numero % 15 == 0:
            coins, ids, _ = detecteur.detectMarkers(image)
            if ids is not None:
                for c, i in zip(coins, ids.ravel()):
                    if i == ID_MARQUEUR:
                        coins_trouves.append(c.reshape(4, 2))
                        dernier_image = image.copy()
        numero += 1
    cap.release()

    if len(coins_trouves) == 0:
        return None, None, 0

    # la mediane evite d'etre perturbe par une detection foireuse
    coins_med = np.median(np.array(coins_trouves), axis=0).astype(np.float32)
    return coins_med, dernier_image, len(coins_trouves)


def main():
    os.makedirs(DOSSIER_CALIB, exist_ok=True)

    for nom in sorted(os.listdir(DOSSIER_VIDEOS)):
        if not nom.lower().endswith((".mp4", ".mov")):
            continue
        base = nom.rsplit(".", 1)[0]

        coins, image, nb = trouver_marqueur(os.path.join(DOSSIER_VIDEOS, nom))
        if coins is None:
            print(nom, ": marqueur NON trouve")
            continue

        # homographie pixels -> metres
        H = cv2.getPerspectiveTransform(coins, COINS_TABLE)
        np.save(os.path.join(DOSSIER_CALIB, base + "_H.npy"), H)

        # taille d'un pixel en mm (juste pour info, varie dans l'image si vue oblique)
        cote_px = np.linalg.norm(coins[1] - coins[0])
        print(nom, ": marqueur trouve sur", nb, "images,",
              "cote =", round(cote_px), "px ->", round(COTE_MARQUEUR * 1000 / cote_px, 2), "mm/px")

        # image de controle avec le contour du marqueur
        cv2.polylines(image, [coins.astype(np.int32)], True, (0, 0, 255), 4)
        petite = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
        cv2.imwrite(os.path.join(DOSSIER_CALIB, base + "_controle.jpg"), petite)


if __name__ == "__main__":
    main()
