# Etape 5 : clonage de comportement (behavior cloning).
# Un petit reseau de neurones (MLP) apprend a reproduire l'action de l'expert
# a partir de l'observation : obs (8 nombres) -> action (4 nombres).

import os
import numpy as np
import torch
import torch.nn as nn

DATASET = "data/dataset_bc.npz"
SORTIE = "modeles/bc.pt"
NB_EPOCHS = 60
TAILLE_BATCH = 256
ECHELLE_ACTION = 0.05  # les deplacements (m) sont divises par 0.05 pour etre autour de [-1, 1]


def creer_reseau():
    return nn.Sequential(
        nn.Linear(8, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 4),
    )


def main():
    torch.manual_seed(0)
    os.makedirs("modeles", exist_ok=True)

    d = np.load(DATASET)
    obs, act, test = d["obs"], d["act"].copy(), d["test"]
    act[:, :3] /= ECHELLE_ACTION

    # on n'apprend que sur les demos d'entrainement
    obs_train, act_train = obs[~test], act[~test]
    moyenne = obs_train.mean(axis=0)
    ecart = obs_train.std(axis=0) + 1e-6

    X = torch.tensor((obs_train - moyenne) / ecart, dtype=torch.float32)
    Y = torch.tensor(act_train, dtype=torch.float32)

    reseau = creer_reseau()
    optim = torch.optim.Adam(reseau.parameters(), lr=1e-3)
    perte_fn = nn.MSELoss()

    for epoch in range(NB_EPOCHS):
        ordre = torch.randperm(len(X))
        total = 0
        for i in range(0, len(X), TAILLE_BATCH):
            idx = ordre[i:i + TAILLE_BATCH]
            perte = perte_fn(reseau(X[idx]), Y[idx])
            optim.zero_grad()
            perte.backward()
            optim.step()
            total += perte.item() * len(idx)
        if epoch % 10 == 0 or epoch == NB_EPOCHS - 1:
            print("epoch", epoch, "perte", round(total / len(X), 5))

    torch.save({"reseau": reseau.state_dict(), "moyenne": moyenne, "ecart": ecart}, SORTIE)
    print("modele sauvegarde dans", SORTIE)


if __name__ == "__main__":
    main()
