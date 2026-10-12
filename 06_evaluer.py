# Etape 6 : evaluer la politique apprise dans la simulation.
# Deux tests :
#   - demos de test : les departs / arrivees des demos gardees de cote (jamais vues)
#   - positions aleatoires : barre et cible tirees au hasard autour de A et B
# On compte le taux de reussite et on enregistre un GIF.

import os
import importlib
import numpy as np
import torch
import imageio
import mujoco
from env_barre import EnvBarre

rejeu = importlib.import_module("04_rejouer_demos")
entrainement = importlib.import_module("05_entrainer_bc")

DOSSIER_DEMOS = "data/demos_robot"
MODELE = "modeles/bc.pt"
NB_PAS_MAX = 400
A_SIM = np.array([0.45, -0.17])
B_SIM = np.array([0.45, 0.18])


def charger_politique(chemin=MODELE):
    sauvegarde = torch.load(chemin, weights_only=False)
    reseau = entrainement.creer_reseau()
    reseau.load_state_dict(sauvegarde["reseau"])
    moyenne, ecart = sauvegarde["moyenne"], sauvegarde["ecart"]

    def politique(obs):
        x = torch.tensor((obs - moyenne) / ecart, dtype=torch.float32)
        with torch.no_grad():
            a = reseau(x).numpy()
        return np.array([*(a[:3] * entrainement.ECHELLE_ACTION), a[3]])
    return politique


def lancer(env, politique, pos_barre, cible, images=None, rendu=None):
    obs = env.reset(pos_barre, cible)
    for k in range(NB_PAS_MAX):
        action = politique(obs)
        action[:3] = np.clip(action[:3], -0.05, 0.05)
        obs = env.step(action)
        if images is not None and k % 4 == 0:
            rendu.update_scene(env.data, camera=camera())
            images.append(rendu.render())
    return env.succes()


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, 0.0, 0.05]
    cam.distance = 1.2
    cam.azimuth = 150
    cam.elevation = -30
    return cam


def main():
    env = EnvBarre()
    politique = charger_politique()
    rng = np.random.default_rng(1)

    # --- test 1 : les demos gardees de cote (1 sur 5, comme dans l'etape 4)
    fichiers = sorted(f for f in os.listdir(DOSSIER_DEMOS) if f.endswith(".npy"))
    resultats = []
    for i, f in enumerate(fichiers):
        if i % 5 != 4:
            continue
        demo = np.load(os.path.join(DOSSIER_DEMOS, f))
        resultats.append(lancer(env, politique, demo[0, 1:], demo[-1, 1:]))
    print("demos de test :", sum(resultats), "/", len(resultats),
          "=", round(np.mean(resultats), 2))

    # --- test 2 : positions aleatoires (+/- 4 cm autour de A et B, dans les deux sens)
    resultats2 = []
    for _ in range(50):
        if rng.random() < 0.5:
            depart, arrivee = A_SIM, B_SIM
        else:
            depart, arrivee = B_SIM, A_SIM
        depart = depart + rng.uniform(-0.04, 0.04, 2)
        arrivee = arrivee + rng.uniform(-0.04, 0.04, 2)
        resultats2.append(lancer(env, politique, depart, arrivee))
    print("positions aleatoires :", sum(resultats2), "/", len(resultats2),
          "=", round(np.mean(resultats2), 2))

    # --- GIF d'un episode
    os.makedirs("resultats", exist_ok=True)
    rendu = mujoco.Renderer(env.model, 360, 480)
    images = []
    lancer(env, politique, A_SIM + [0.01, 0.01], B_SIM, images, rendu)
    imageio.mimsave("resultats/politique_bc.gif", images, duration=0.06)
    print("GIF : resultats/politique_bc.gif")


if __name__ == "__main__":
    main()
