# Etape 4 : faire rejouer les demos humaines par le robot (expert scripte).
# Pour chaque demo retargetee :
#   1. le robot va au-dessus de la barre (a l'endroit ou l'humain l'a prise)
#   2. il descend, ferme la pince, souleve
#   3. il suit le trajet de la main humaine (x, y) jusqu'a l'arrivee
#   4. il descend, ouvre la pince, remonte
# On enregistre (observation, action) a chaque pas : ce sera le jeu de donnees du clonage
# de comportement (etape 5). On garde 20 % des demos de cote pour tester.

import os
import numpy as np
from env_barre import EnvBarre

DOSSIER_DEMOS = "data/demos_robot"
SORTIE = "data/dataset_rejeu.npz"
DT = 0.02                 # duree d'un pas de commande (s)
Z_HAUT = 0.15             # hauteur pour aller au-dessus de la barre
Z_PRISE = 0.02            # hauteur de la pince pour saisir
Z_TRANSPORT = 0.10        # hauteur pendant le transport
NB_COPIES = 3             # chaque demo est rejouee avec de petits decalages aleatoires
DECALAGE = 0.02           # ecart-type du decalage (m)


def interpoler(p_debut, p_fin, n):
    """n points regulierement espaces entre deux positions 3D (sans le point de depart)."""
    return [p_debut + (p_fin - p_debut) * (k + 1) / n for k in range(n)]


def construire_plan(demo, decalage):
    """Liste de (position voulue de la pince xyz, pince ouverte ?) pour chaque pas."""
    xy = demo[:, 1:] + decalage
    depart, arrivee = xy[0], xy[-1]
    plan = []

    # position de la pince au debut = a peu pres celle du reset
    courante = np.array([0.6, 0.0, 0.2])

    def aller(cible, n, ouvert):
        nonlocal courante
        for p in interpoler(courante, cible, n):
            plan.append((p, ouvert))
        courante = cible

    aller(np.array([*depart, Z_HAUT]), 40, True)
    aller(np.array([*depart, Z_PRISE]), 40, True)
    aller(np.array([*depart, Z_PRISE]), 15, True)       # on attend que la pince soit bien arrivee
    aller(np.array([*depart, Z_PRISE]), 20, False)      # fermeture
    aller(np.array([*depart, Z_TRANSPORT]), 35, False)

    # transport : on suit le trajet humain, reechantillonne au rythme de 20 ms
    n_transport = max(int(demo[-1, 0] / DT), 10)
    temps = np.linspace(0, demo[-1, 0], n_transport)
    for t in temps:
        x = np.interp(t, demo[:, 0], xy[:, 0])
        y = np.interp(t, demo[:, 0], xy[:, 1])
        plan.append((np.array([x, y, Z_TRANSPORT]), False))
    courante = np.array([*arrivee, Z_TRANSPORT])

    aller(np.array([*arrivee, Z_PRISE + 0.01]), 35, False)
    aller(np.array([*arrivee, Z_PRISE + 0.01]), 20, True)  # ouverture
    aller(np.array([*arrivee, Z_HAUT]), 20, True)
    return plan, depart, arrivee


def rejouer(env, demo, decalage=np.zeros(2), enregistrer=True):
    plan, depart, arrivee = construire_plan(demo, decalage)
    obs = env.reset(depart, arrivee)
    observations, actions = [], []

    for position_voulue, ouvert in plan:
        delta = position_voulue - env.consigne
        delta = np.clip(delta, -0.05, 0.05)
        action = np.array([*delta, 1.0 if ouvert else -1.0])
        if enregistrer:
            observations.append(obs)
            actions.append(action)
        obs = env.step(action)

    return env.succes(), np.array(observations), np.array(actions)


def main():
    env = EnvBarre()
    rng = np.random.default_rng(0)
    fichiers = sorted(f for f in os.listdir(DOSSIER_DEMOS) if f.endswith(".npy"))

    # 1 demo sur 5 sert de test (on prend les indices 4, 9, 14...)
    obs_tout, act_tout, ep_tout, est_test = [], [], [], []
    succes_train, succes_test = [], []

    for i, f in enumerate(fichiers):
        demo = np.load(os.path.join(DOSSIER_DEMOS, f))
        test = (i % 5 == 4)
        copies = 1 if test else NB_COPIES

        for c in range(copies):
            decalage = np.zeros(2) if (test or c == 0) else rng.normal(0, DECALAGE, 2)
            ok, obs, act = rejouer(env, demo, decalage)
            (succes_test if test else succes_train).append(ok)
            obs_tout.append(obs)
            act_tout.append(act)
            ep_tout.append(np.full(len(obs), len(ep_tout)))
            est_test.append(np.full(len(obs), test))
        if i % 10 == 0:
            print(i, "/", len(fichiers), "demos rejouees")

    print("succes du rejeu : train", round(np.mean(succes_train), 2),
          "test", round(np.mean(succes_test), 2))
    np.savez(SORTIE,
             obs=np.concatenate(obs_tout), act=np.concatenate(act_tout),
             episode=np.concatenate(ep_tout), test=np.concatenate(est_test))
    print("dataset :", sum(len(o) for o in obs_tout), "pas")


if __name__ == "__main__":
    main()
