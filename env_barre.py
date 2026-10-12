# Simulation MuJoCo : un Panda doit deplacer une barre de A vers B.
# L'action du robot = un deplacement de la pince (dx, dy, dz) + ouvrir/fermer.
# Une cinematique inverse simple transforme la position voulue en angles des 7 articulations.

import numpy as np
import mujoco

CHEMIN_SCENE = "robot/franka_emika_panda/scene_barre.xml"

LONGUEUR_PINCE = 0.10      # distance entre l'origine de la main et le bout des doigts
Z_BARRE = 0.012            # hauteur du centre de la barre posee sur la table
# orientation voulue de la main : doigts vers le bas, la pince se ferme selon l'axe y du monde
R_VOULUE = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)

POSE_DEPART = np.array([0, 0.3, 0, -2.0, 0, 2.3, -0.785])  # position initiale des 7 articulations


class EnvBarre:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(CHEMIN_SCENE)
        self.data = mujoco.MjData(self.model)
        self.id_main = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.id_barre = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barre")
        self.cible = np.zeros(2)       # position d'arrivee voulue (x, y)
        self.pince_ouverte = True
        # la pince de base serre trop faiblement : on multiplie sa raideur par 10
        self.model.actuator_gainprm[7, 0] *= 10
        self.model.actuator_biasprm[7, 1] *= 10
        self.nb_pas_sim = 10           # pas de simulation par action (10 x 2 ms = 20 ms)

    # ---------- utilitaires ----------
    def position_pince(self):
        """Position du bout des doigts dans le monde."""
        R = self.data.xmat[self.id_main].reshape(3, 3)
        return self.data.xpos[self.id_main] + R[:, 2] * LONGUEUR_PINCE

    def position_barre(self):
        return self.data.xpos[self.id_barre].copy()

    def observation(self):
        """Ce que voit la politique : pince (xyz), ouverture, barre (xy), cible (xy)."""
        ouverture = self.data.qpos[7] * 2  # largeur entre les doigts (m)
        return np.concatenate([self.position_pince(), [ouverture],
                               self.position_barre()[:2], self.cible]).astype(np.float32)

    # ---------- cinematique inverse ----------
    def angles_pour(self, position_voulue):
        """Methode des moindres carres amortis : on corrige les angles petit a petit
        jusqu'a ce que le bout de la pince soit a la bonne position avec la bonne orientation."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(30):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)  # necessaire pour que le jacobien soit a jour
            R = d.xmat[self.id_main].reshape(3, 3)
            bout = d.xpos[self.id_main] + R[:, 2] * LONGUEUR_PINCE

            erreur_pos = position_voulue - bout
            # erreur d'orientation (petits angles) : somme des produits vectoriels des axes
            erreur_rot = 0.5 * sum(np.cross(R[:, i], R_VOULUE[:, i]) for i in range(3))
            if np.linalg.norm(erreur_pos) < 1e-4 and np.linalg.norm(erreur_rot) < 1e-3:
                break

            mujoco.mj_jac(self.model, d, jacp, jacr, bout, self.id_main)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            e = np.concatenate([erreur_pos, erreur_rot])
            # inverse amortie : evite les mouvements brusques pres des singularites
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d.qpos[:7] += dq
        return d.qpos[:7].copy()

    # ---------- interface type gym ----------
    def reset(self, pos_barre, cible):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = POSE_DEPART
        self.data.qpos[7:9] = 0.04
        self.data.ctrl[:7] = POSE_DEPART
        self.data.ctrl[7] = 255
        # position de la barre (le free joint commence apres les 9 articulations du bras)
        self.data.qpos[9:12] = [pos_barre[0], pos_barre[1], Z_BARRE]
        self.data.qpos[12:16] = [1, 0, 0, 0]
        self.cible = np.array(cible, dtype=float)
        self.pince_ouverte = True
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.consigne = self.position_pince().copy()
        return self.observation()

    def aller_a(self, position_voulue, ouvrir):
        """Envoie la pince vers une position absolue (x, y, z)."""
        self.consigne = np.array(position_voulue, dtype=float)
        self.data.ctrl[:7] = self.angles_pour(position_voulue)
        self.data.ctrl[7] = 255 if ouvrir else 0
        self.pince_ouverte = ouvrir
        for _ in range(self.nb_pas_sim):
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def step(self, action):
        """action = (dx, dy, dz, pince) : deplacement en metres, pince > 0 = ouverte.
        Le deplacement est ajoute a la consigne (et non a la position reelle) : sinon la
        pince, qui retombe un peu sous son propre poids, n'arrive jamais a destination."""
        voulue = self.consigne + np.array(action[:3])
        voulue[2] = max(voulue[2], 0.012)   # on ne descend pas dans la table
        return self.aller_a(voulue, action[3] > 0)

    def succes(self):
        """La barre est posee pres de la cible (a moins de 3 cm, et sur la table)."""
        p = self.position_barre()
        return np.linalg.norm(p[:2] - self.cible) < 0.03 and p[2] < 0.03
