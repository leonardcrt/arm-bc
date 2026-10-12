# Etape 4b : un expert "reactif" pour generer les donnees du clonage de comportement.
# Probleme du rejeu (etape 4) : il depend du temps ecoule (on attend 15 pas, puis on ferme...),
# et un reseau qui ne voit que la scene ne sait pas ou il en est -> il reste bloque.
# Ici l'expert decide uniquement a partir de l'observation (position de la pince,
# ouverture, position de la barre, cible), comme le fera la politique apprise.
#
# Ce qui vient des demos humaines :
#   - les positions de depart et d'arrivee (on les tire dans les vraies demos)
#   - le profil de vitesse pendant le transport (l'humain ralentit en approchant de la cible)

import os
import numpy as np
from env_barre import EnvBarre

DOSSIER_DEMOS = "data/demos_robot"
SORTIE = "data/dataset_bc.npz"

Z_HAUT = 0.15
Z_TRANSPORT = 0.10
Z_PRISE = 0.02
Z_RELACHE = 0.025
NB_PAS_MAX = 450


def profil_vitesse():
    """Vitesse moyenne de l'humain (m par pas de 20 ms) en fonction de la distance restante.
    Retourne deux tableaux (distances, vitesses) pour np.interp."""
    bords = np.arange(0, 0.45, 0.02)
    somme = np.zeros(len(bords))
    compte = np.zeros(len(bords))
    for f in sorted(os.listdir(DOSSIER_DEMOS)):
        d = np.load(os.path.join(DOSSIER_DEMOS, f))
        xy = d[:, 1:]
        dt = np.gradient(d[:, 0])
        vitesse = np.linalg.norm(np.gradient(xy, axis=0), axis=1) / dt * 0.02
        restant = np.linalg.norm(xy - xy[-1], axis=1)
        # on ne garde que la phase de mouvement (vitesse non nulle)
        for r, v in zip(restant, vitesse):
            k = int(r / 0.02)
            if k < len(bords) and v > 0.0005:
                somme[k] += v
                compte[k] += 1
    ok = compte > 5
    return bords[ok] + 0.01, somme[ok] / compte[ok]


def expert(obs, profil):
    """obs = [pince x,y,z, ouverture, barre x,y, cible x,y]. Retourne (dx, dy, dz, pince)."""
    pince, ouv = obs[:3], obs[3]
    barre, cible = obs[4:6], obs[6:8]
    # la hauteur de la barre n'est pas dans l'observation : on la deduit de la pince
    # quand elle la tient (la barre est a ~1 cm sous la pince)
    arrivee = np.linalg.norm(barre - cible) < 0.02

    # 1) barre arrivee : on ouvre la pince puis on remonte
    if arrivee and pince[2] < Z_RELACHE + 0.02:
        if ouv > 0.05:
            return np.array([0, 0, 0.01, 1.0])      # deja ouverte : on remonte
        return np.array([0, 0, 0, 1.0])             # on attend l'ouverture
    if arrivee and ouv > 0.06:
        return np.array([0, 0, 0.01 if pince[2] < Z_HAUT else 0, 1.0])

    ecart_barre = barre - pince[:2]
    dist_barre = np.linalg.norm(ecart_barre)
    tient = 0.02 < ouv < 0.045 and dist_barre < 0.03

    # 2) la barre est dans la pince : transport vers la cible
    if tient:
        reste = cible - pince[:2]
        d = np.linalg.norm(reste)
        if d > 0.012:
            if pince[2] < Z_TRANSPORT - 0.02 and d > 0.03:
                return np.array([0, 0, 0.01, -1.0])   # d'abord on souleve
            v = min(np.interp(d, *profil), d)
            xy = reste / d * v
            dz = np.clip(Z_TRANSPORT - pince[2], -0.01, 0.01)
            return np.array([xy[0], xy[1], dz, -1.0])
        # au-dessus de la cible : on descend
        xy = np.clip(reste * 0.5, -0.005, 0.005)
        return np.array([xy[0], xy[1], -0.008, -1.0])

    # 3) pince en train de se fermer
    if 0.02 <= ouv <= 0.06 and pince[2] < 0.03:
        return np.array([0, 0, 0, -1.0])
    # fermee dans le vide : on reouvre
    if ouv < 0.02:
        return np.array([0, 0, 0.01, 1.0])

    # 4) pince ouverte : aller chercher la barre
    if dist_barre > 0.02 or (dist_barre > 0.008 and pince[2] > 0.06):
        # on se deplace en gardant de la hauteur
        if dist_barre > 0.02 and pince[2] < 0.08:
            return np.array([0, 0, 0.01, 1.0])
        xy = np.clip(ecart_barre * 0.5, -0.01, 0.01)
        dz = np.clip(Z_HAUT - pince[2], -0.01, 0.01) if dist_barre > 0.02 else \
            np.clip(0.07 - pince[2], -0.01, 0.01)
        return np.array([xy[0], xy[1], dz, 1.0])

    # aligne : on descend, puis on ferme
    xy = np.clip(ecart_barre * 0.5, -0.003, 0.003)
    if pince[2] > Z_PRISE + 0.004:
        return np.array([xy[0], xy[1], np.clip(Z_PRISE - pince[2], -0.01, 0), 1.0])
    return np.array([0, 0, 0, -1.0])


def episode(env, profil, depart, arrivee):
    obs = env.reset(depart, arrivee)
    observations, actions = [], []
    reussi_depuis = 0
    for _ in range(NB_PAS_MAX):
        action = expert(obs, profil)
        observations.append(obs)
        actions.append(action)
        obs = env.step(action)
        # on s'arrete un peu apres la reussite (le temps de remonter la pince)
        reussi_depuis = reussi_depuis + 1 if env.succes() and obs[2] > 0.1 else 0
        if reussi_depuis > 3:
            break
    return env.succes(), np.array(observations), np.array(actions)


def main():
    env = EnvBarre()
    profil = profil_vitesse()
    print("profil de vitesse (distance restante -> m/pas) :",
          np.round(profil[0], 2), np.round(profil[1], 4))
    rng = np.random.default_rng(0)

    fichiers = sorted(f for f in os.listdir(DOSSIER_DEMOS) if f.endswith(".npy"))
    demos = [np.load(os.path.join(DOSSIER_DEMOS, f)) for f in fichiers]

    obs_tout, act_tout, test_tout = [], [], []
    succes = {"train": [], "test": []}

    # --- entrainement : departs/arrivees des demos d'entrainement, avec des decalages aleatoires
    train = [d for i, d in enumerate(demos) if i % 5 != 4]
    for n in range(400):
        d = train[rng.integers(len(train))]
        depart = d[0, 1:] + rng.normal(0, 0.02, 2)
        arrivee = d[-1, 1:] + rng.normal(0, 0.02, 2)
        ok, o, a = episode(env, profil, depart, arrivee)
        succes["train"].append(ok)
        obs_tout.append(o)
        act_tout.append(a)
        test_tout.append(np.zeros(len(o), dtype=bool))
        if n % 50 == 0:
            print(n, "/ 400 episodes, succes", round(np.mean(succes["train"]), 2))

    # --- test : departs/arrivees exacts des demos gardees de cote
    for i, d in enumerate(demos):
        if i % 5 != 4:
            continue
        ok, o, a = episode(env, profil, d[0, 1:], d[-1, 1:])
        succes["test"].append(ok)
        obs_tout.append(o)
        act_tout.append(a)
        test_tout.append(np.ones(len(o), dtype=bool))

    print("succes de l'expert : train", round(np.mean(succes["train"]), 2),
          "test", round(np.mean(succes["test"]), 2))
    np.savez(SORTIE, obs=np.concatenate(obs_tout), act=np.concatenate(act_tout),
             test=np.concatenate(test_tout))
    print("dataset :", sum(len(o) for o in obs_tout), "pas")


if __name__ == "__main__":
    main()
