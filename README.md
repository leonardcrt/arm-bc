# Piloter un bras robot simule avec mes propres videos

Projet pour le challenge stagiaire Humanoid. Les donnees sont des videos que j'ai filmees
moi-meme au telephone : je deplace une barre de cereales entre deux endroits sur mon bureau,
une vingtaine de fois, avec un marqueur ArUco (10 cm) pose sur la table pour avoir une echelle.

(README a completer : resultats, choix de conception, ce qui a marche / pas marche.)

## Pipeline

| Etape | Script | Ce que ca fait |
|---|---|---|
| 0 | `00_calibration.py` | detecte le marqueur ArUco et calcule la transformation pixels -> metres sur la table |
| 1 | `01_suivre_objet.py` | suit la barre (couleur verte) image par image et la convertit en metres |
| 2 | `02_decouper.py` | decoupe la longue video en demos (un aller de A vers B, ou un retour) |

Les etapes suivantes (robot simule, politique apprise) sont a venir.

## Lancer

```
pip install -r requirements.txt
# mettre les videos .mp4 / .mov dans data/videos/
python 00_calibration.py
python 01_suivre_objet.py
python 02_decouper.py
```

Les videos ne sont pas dans le depot (trop lourdes). Les positions suivies
(`data/suivi/`) et les demos (`data/demos/`) y sont.

## Organisation

- `data/videos/` : mes videos (non versionnees)
- `data/calibration/` : homographie de chaque video + image de controle
- `data/suivi/` : position de l'objet image par image
- `data/demos/` : une trajectoire par demonstration
- `resultats/` : graphiques de controle, GIFs
