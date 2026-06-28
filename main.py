import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib import font_manager as fm
import matplotlib.image as mpimg
import matplotlib.patheffects as pe

# ============================================================
# PARAMÈTRES
# ============================================================
LATITUDE = 45
MAX_MAGNITUDE = 4.5
LABEL_MAG_LIMIT = 1  # seuil pour afficher le nom des étoiles
STAR_LABEL_GAP_PT = 2  # espace visuel constant entre le bord du marqueur et son libellé
LABEL_COLLISION_PADDING_PX = 0
STAR_DOT_COLLISION_PADDING_PX = 1.0
CONST_LABEL_AVOID_STAR_DOTS = False  # si False, les labels de constellations peuvent chevaucher les étoiles
LABEL_DISK_MARGIN = 6  # marge en unités carte pour garder les labels dans le disque

# Traits de rappel label → objet
DRAW_LEADER_LINES = True
LEADER_LINE_LW = 0.4
LEADER_LINE_ALPHA = 0.55
LEADER_STAR_MIN_STEP_PT = 4.0      # distance min (pt) à partir de laquelle tracer le trait
LEADER_CONST_MIN_DU = 3.0          # distance min (unités carte) pour les constellations

# Constellations
CONSTELLATIONS_FILE = "constellations_fr.json"
DRAW_CONSTELLATION_BOUNDARIES = False
DRAW_CONSTELLATION_LABELS = True

# ============================================================
# FORÇAGE DES LABELS
# ============================================================
# Constellations : labels toujours affichés (même si collision avec des dots d'étoiles)
FORCED_CONSTELLATION_NAMES = [
    "Hercule",
    "Bouvier",
    "Couronne\nBoréale",
    "Poissons",
    "Gémeaux",
    "Tête du\nSerpent",
    "Queue du\nSerpent",
    "Cygne",
    "Grand Chien",
]

# Étoiles : labels toujours affichés (même si collision avec un dot d'étoile)
# -> Mets ici les noms affichés (ex: "Polaris", "Sirius", ...)
FORCED_STAR_NAMES = [
    "Polaris",
    "Arcturus",
    "Spica",
    "Capella",
]

# Clipping
CLIP_SKY = True
CLIP_DATES = False  # si True, mois/jours clippés au disque

# Page 2 (masque simple)
MASK_COLOR = "#000000"  # disque plein
PAPER_COLOR = "white"   # couleur simulant la transparence (papier blanc)
DRAW_HORIZON_OUTLINE_ON_MASK = True
NAMED_STAR_COLOR = "#FF0000"
STAR_DEFAULT_COLOR = "black"
CONST_LINE_COLOR = "#444444"
CONST_LABEL_COLOR = "#333333"
CONSTELLATION_LINE_ZORDER = 10
STAR_DOT_ZORDER = 20

# Voie Lactée
MW_FILE = "mw.json"
DRAW_MILKY_WAY = True
MW_COLOR = "#b8c8d8"   # bleu-gris clair pour impression sur blanc
MW_ALPHA_MIN = 0.35    # couche externe (ol1)
MW_ALPHA_MAX = 0.55    # couche interne / noyau galactique (ol5)

# Trait N-S à travers la fenêtre (visible uniquement dans la fenêtre blanche)
DRAW_VERTICAL_WINDOW_LINE = True
WINDOW_LINE_WIDTH = 1.0

# Cardinaux sur la courbe d'horizon (bord intérieur), tangents à la courbe
DRAW_CARDINALS_ON_HORIZON = True
CARDINAL_FONT = 8
CARDINAL_OFFSET = 5  # décalage radial (unités de la carte) vers l’extérieur

# Texte en haut du masque d'horizon (page 2), le long de la courbe
DRAW_HORIZON_TOP_TEXT = True
HORIZON_TOP_TEXT = "Carte visible pour les latitudes proches de" + f" {LATITUDE}°N ---- Heure TU - Retirer 2h en été et 1h en hiver"
HORIZON_TOP_TEXT_PADDING = 5
HORIZON_TOP_TEXT_FONT = 5
HORIZON_TOP_TEXT_CHAR_STEP_DEG = 0.8  # espacement angulaire entre les caractères (en degrés sur la courbe)

# Marqueur du zénith (centre du ciel visible dans la fenêtre d'horizon)
DRAW_ZENITH_MARKER = True
ZENITH_MARKER_RADIUS = 1.2
ZENITH_MARKER_COLOR = "#FF0000"

# Petite croix rouge au centre du cercle externe (page 2)
DRAW_PAGE2_CENTER_CROSS = True
PAGE2_CENTER_CROSS_HALF_SIZE = 1.2
PAGE2_CENTER_CROSS_LW = 1.0
PAGE2_CENTER_CROSS_COLOR = "#FF0000"

# ============================================================
# POLICE PERSONNALISÉE
# ============================================================
GILROY_BLACK_FONT_PATH = "fonts/Gilroy-Medium.ttf"
GILROY_MEDIUM_FONT_PATH = "fonts/Gilroy-Medium.ttf"
DMMONO_FONT_PATH = "fonts/DMMono-Medium.ttf"

gilroy_black = fm.FontProperties(fname=GILROY_BLACK_FONT_PATH)
gilroy_medium = fm.FontProperties(fname=GILROY_MEDIUM_FONT_PATH)
dmmono = fm.FontProperties(fname=DMMONO_FONT_PATH)

# Optionnel : enregistrer la police globalement
fm.fontManager.addfont(GILROY_BLACK_FONT_PATH)
fm.fontManager.addfont(GILROY_MEDIUM_FONT_PATH)
fm.fontManager.addfont(DMMONO_FONT_PATH)

# ------------------------------------------------------------
# Mise en page A4 / marges d'impression
# ------------------------------------------------------------
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
CM_PER_INCH = 2.54
PRINT_MARGIN_CM = 0.5


# ============================================================
# LOGO (page 2)
# ============================================================
LOGO_PATH = "assets/LOGO_WHITE.png"
LOGO_SIZE = 0.20         # taille relative (0.1 = petit, 0.2 = grand)
LOGO_X = 0.50            # position horizontale (0 gauche → 1 droite)
LOGO_Y = 0.65            # position verticale (0 bas → 1 haut)

# ------------------------------------------------------------
# Bandes des anneaux (multiples de max_radius)
# ------------------------------------------------------------
# Zone ROSE (page 1) : mois + jours
PINK_IN = 1
PINK_OUT = 1.13

# Zone BLEUE (page 2) : heures (au-dessus de la zone rose)
BLUE_IN = 1.13
BLUE_OUT = 1.23

# Paddings radiaux (en "unités carte" – mêmes unités que max_radius, donc ~degrés)
# -> Permet de poser les textes sur les "lignes" intérieure/extérieure des bandes
TEXT_PAD_IN = 3.5   # distance depuis la bordure intérieure (vers l'extérieur)
TEXT_PAD_OUT = 3.5  # distance depuis la bordure extérieure (vers l'intérieur)

# Ticks jours (zone rose) : du bord extérieur vers l'intérieur
DAY_TICK_OUT_PAD = 5.5
DAY_TICK_LEN = 1.8
DAY_MARKER_RADIUS = 0.6

# Séparateurs de mois sur l'anneau dates/mois
DRAW_MONTH_SEPARATORS = True
MONTH_SEPARATOR_COLOR = "white"
MONTH_SEPARATOR_LW = 0.5
MONTH_SEPARATOR_OUT_PAD = 3
MONTH_SEPARATOR_IN_PAD = 3

# Ticks heures (zone bleue) : du bord extérieur vers l'intérieur
HOUR_TICK_OUT_PAD = 0.2
HOUR_TICK_LEN_15 = 2.6
HOUR_TICK_LEN_30 = 3.2
HOUR_TICK_LEN_60 = 3.9

# Heures (page 2) : labels 30 min
HOUR_LABEL_FONT = 7

# Cercle de découpe (commun aux 2 pages)
DRAW_CUT_CIRCLE = True
CUT_RADIUS_FACTOR = BLUE_OUT  # on découpe au bord externe de la zone bleue
CUT_CIRCLE_LW = 1.0
CUT_CIRCLE_STYLE = "-"  # "--" ou ":" si tu veux pointillé
CUT_CIRCLE_COLOR = "black"

# ------------------------------------------------------------
# Couleurs des zones (optionnel, correspond à ton schéma)
# ------------------------------------------------------------
DRAW_PINK_BAND = True
DRAW_BLUE_BAND = True
PINK_COLOR = "#000000"
BLUE_COLOR = "#FFFFFF"
BAND_ALPHA = 1.0

# Plus grand carré imprimable avec marge mini de 1.5 cm
MAP_SIDE_CM = min(
    A4_WIDTH_CM - (2.0 * PRINT_MARGIN_CM),
    A4_HEIGHT_CM - (2.0 * PRINT_MARGIN_CM),
)

MARGIN_WIDTH = MAP_SIDE_CM / A4_WIDTH_CM
MARGIN_HEIGHT = MAP_SIDE_CM / A4_HEIGHT_CM
MARGIN_LEFT = (1.0 - MARGIN_WIDTH) / 2.0
MARGIN_BOTTOM = (1.0 - MARGIN_HEIGHT) / 2.0

FIGURE_SIZE_A4_IN = (A4_WIDTH_CM / CM_PER_INCH, A4_HEIGHT_CM / CM_PER_INCH)

# Rayon max extérieur à afficher (doit inclure anneaux + taille de police)
OUTER_RING_FACTOR = 1.26

# Titres en haut, hors zone utile
TITLE_Y = 0.96

phi = np.radians(LATITUDE)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
with open("stars.json", "r", encoding="utf-8") as f:
    data = json.load(f)
stars = data["data"]


def load_constellations(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return raw.get("data", [])
        return []
    except FileNotFoundError:
        print(f"[WARN] Fichier constellations introuvable: {path} (constellations ignorées)")
        return []


constellations = load_constellations(CONSTELLATIONS_FILE)
print(f"Constellations chargées : {len(constellations)}")

# ============================================================
# HELPERS
# ============================================================
def is_visible_star(dec, latitude=LATITUDE):
    return dec > (latitude - 90)


visible_stars = [
    star for star in stars
    if is_visible_star(star["dec"]) and star["V"] <= MAX_MAGNITUDE
]


def extract_star_name(ids_string):
    if not ids_string:
        return None
    parts = str(ids_string).split("|")
    for p in parts:
        if p.startswith("NAME "):
            return p.replace("NAME ", "").strip()
    return None


def project_star(ra_deg, dec_deg):
    # Repère actuel : H = -RA
    H = np.radians(-ra_deg)
    r = 90 - dec_deg
    x = r * np.sin(H)
    y = -r * np.cos(H)
    return x, y


def unwrap_ra_pair(ra1, ra2):
    d = ra2 - ra1
    if d > 180:
        ra2 -= 360
    elif d < -180:
        ra2 += 360
    return ra1, ra2


def get_tangent_rotation(x, y):
    angle = np.degrees(np.arctan2(y, x))
    return angle + 90


def clip_artist(artist, clip_patch):
    if artist is None or clip_patch is None:
        return artist
    try:
        artist.set_clip_path(clip_patch)
    except Exception:
        pass
    return artist


def is_valid_coord(ra, dec):
    return ra is not None and dec is not None


def is_inside_disk_xy(x, y, R):
    return (x * x + y * y) <= (R * R)


def marker_radius_points(area_points2):
    return np.sqrt(max(area_points2, 0.0) / np.pi)


def data_offset_from_points(ax, x, y, dx_pt=0.0, dy_pt=0.0):
    px_per_pt = ax.figure.dpi / 72.0
    x_px, y_px = ax.transData.transform((x, y))
    tx_px = x_px + dx_pt * px_per_pt
    ty_px = y_px + dy_pt * px_per_pt
    return ax.transData.inverted().transform((tx_px, ty_px))


def screen_basis_vectors(ax, x, y):
    """Retourne (radial_unit_px, tangential_unit_px) en coordonnées écran (pixels)."""
    x_px, y_px = ax.transData.transform((x, y))
    r = float(np.hypot(x, y))
    if r == 0.0:
        # au centre : radial vers le haut de la carte (nord)
        x2, y2 = (0.0, -1.0)
    else:
        x2, y2 = (x / r, y / r)

    x2_px, y2_px = ax.transData.transform((x + x2, y + y2))
    vx = float(x2_px - x_px)
    vy = float(y2_px - y_px)
    n = math.hypot(vx, vy) or 1.0
    rx, ry = vx / n, vy / n  # radial outward

    # tangentiel (sens trigonométrique)
    tx, ty = -ry, rx
    return (rx, ry), (tx, ty)


def data_offset_from_points_basis(ax, x, y, radial_pt=0.0, tangential_pt=0.0):
    """Décale un point (x,y) d'une distance en points, exprimée en base (radial/tangentiel) écran."""
    px_per_pt = ax.figure.dpi / 72.0
    (rx, ry), (tx, ty) = screen_basis_vectors(ax, x, y)

    x_px, y_px = ax.transData.transform((x, y))
    dx_px = (radial_pt * rx + tangential_pt * tx) * px_per_pt
    dy_px = (radial_pt * ry + tangential_pt * ty) * px_per_pt

    return ax.transData.inverted().transform((x_px + dx_px, y_px + dy_px))


def outward_space_points(ax, x, y, disk_R):
    """Espace disponible vers l'extérieur du disque, en points, le long de la direction radiale."""
    r = float(np.hypot(x, y))
    if disk_R is None:
        return float("inf")
    if r >= disk_R:
        return 0.0

    # point sur le bord du disque dans la direction radiale
    if r == 0.0:
        xb, yb = (0.0, -disk_R)
    else:
        xb, yb = (x * disk_R / r, y * disk_R / r)

    (rx, ry), _ = screen_basis_vectors(ax, x, y)
    x_px, y_px = ax.transData.transform((x, y))
    xb_px, yb_px = ax.transData.transform((xb, yb))

    # projection sur le vecteur radial écran
    proj_px = (xb_px - x_px) * rx + (yb_px - y_px) * ry
    pt_per_px = 72.0 / ax.figure.dpi
    return max(0.0, proj_px * pt_per_px)


def offset_tangential(x, y, delta):
    r = np.hypot(x, y)
    if r == 0:
        return x, y
    tx, ty = -y / r, x / r
    return x + tx * delta, y + ty * delta


def padded_bbox(bbox, pad_px=0.0):
    if pad_px <= 0:
        return bbox
    return Bbox.from_extents(
        bbox.x0 - pad_px,
        bbox.y0 - pad_px,
        bbox.x1 + pad_px,
        bbox.y1 + pad_px,
    )


def register_text_if_no_overlap(ax, text_artist, occupied_bboxes, renderer=None, pad_px=0.0, register_bbox=True, extra_bboxes=None):
    if occupied_bboxes is None:
        return True

    if renderer is None:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()

    bbox = padded_bbox(text_artist.get_window_extent(renderer=renderer), pad_px=pad_px)
    for used in (occupied_bboxes or []):
        if bbox.overlaps(used):
            text_artist.remove()
            return False

    for used in (extra_bboxes or []):
        if bbox.overlaps(used):
            text_artist.remove()
            return False

    if register_bbox:
        occupied_bboxes.append(bbox)
    return True


def place_star_label(ax, x, y, label, marker_area_points2, occupied_bboxes, renderer, extra_bboxes=None,
                     clip_patch=None, disk_R=None, fontsize=5, collision_pad_px=0.0, color="black"):
    """
    Placement anti-chevauchement pour les labels d'étoiles, en restant le plus proche possible du point de base.
    - priorité aux étoiles : appelé avant les constellations.
    - évite l'effet "plus près du bord => plus loin" en testant d'abord des positions compatibles avec l'espace
      disponible vers l'extérieur (si manque de place, on privilégie l'intérieur du disque).
    """
    if not label:
        return None

    base_offset_pt = marker_radius_points(marker_area_points2) + STAR_LABEL_GAP_PT
    rot = get_tangent_rotation(x, y)

    # si proche du bord, on privilégie l'intérieur (radial négatif)
    out_space_pt = outward_space_points(ax, x, y, disk_R)
    prefer_inward = out_space_pt < (base_offset_pt + 1.0)

    # candidates exprimés en (radial_sign, tangential_sign)
    directions = [
        (1.0, 0.0),   # extérieur
        (0.0, 1.0),   # tangentiel +
        (0.0, -1.0),  # tangentiel -
        (-1.0, 0.0),  # intérieur
        (0.7, 0.7),
        (0.7, -0.7),
        (-0.7, 0.7),
        (-0.7, -0.7),
    ]
    if prefer_inward:
        # on met l'intérieur en premier
        directions = [(-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0),
                      (-0.7, 0.7), (-0.7, -0.7), (0.7, 0.7), (0.7, -0.7)]

    distance_steps_pt = [0.0, 1.5, 3.0, 5.0, 7.0]

    for step in distance_steps_pt:
        dist = base_offset_pt + step
        for rs, ts in directions:
            tx, ty = data_offset_from_points_basis(ax, x, y, radial_pt=rs * dist, tangential_pt=ts * dist)

            if disk_R is not None and not is_inside_disk_xy(tx, ty, disk_R):
                continue

            t = ax.text(
                tx, ty, label,
                fontsize=fontsize,
                color=color,
                rotation=rot,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontproperties=gilroy_black,
                zorder=80,
            )
            t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
            clip_artist(t, clip_patch)

            ok = register_text_if_no_overlap(
                ax,
                t,
                occupied_bboxes=occupied_bboxes,
                renderer=renderer,
                pad_px=collision_pad_px,
                extra_bboxes=extra_bboxes,
                register_bbox=True,
            )
            if ok:
                if DRAW_LEADER_LINES and step > LEADER_STAR_MIN_STEP_PT:
                    leader, = ax.plot([x, tx], [y, ty],
                                      lw=LEADER_LINE_LW, color=color,
                                      alpha=LEADER_LINE_ALPHA, zorder=79,
                                      solid_capstyle="round")
                    clip_artist(leader, clip_patch)
                return t

    return None


def star_marker_style(star):
    size = max(1, 2.5 - star["V"]) ** 2
    name = extract_star_name(star.get("ids"))
    is_labeled = bool(name and (name == "Lodestar" or star["V"] <= LABEL_MAG_LIMIT))
    return name, size, NAMED_STAR_COLOR if is_labeled else STAR_DEFAULT_COLOR


def marker_bbox_from_data(ax, x, y, area_points2, pad_px=0.0):
    radius_px = marker_radius_points(area_points2) * (ax.figure.dpi / 72.0)
    x_px, y_px = ax.transData.transform((x, y))
    bbox = Bbox.from_extents(
        x_px - radius_px,
        y_px - radius_px,
        x_px + radius_px,
        y_px + radius_px,
    )
    return padded_bbox(bbox, pad_px=pad_px)


def setup_map_ax(fig, outer_r):
    ax = fig.add_axes([MARGIN_LEFT, MARGIN_BOTTOM, MARGIN_WIDTH, MARGIN_HEIGHT])
    ax.set_xlim(-outer_r, outer_r)
    ax.set_ylim(-outer_r, outer_r)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def add_annulus(ax, r_in, r_out, color, alpha=1.0, zorder=0):
    """
    Dessine une couronne (anneau) simple :
    - cercle extérieur rempli (couleur)
    - cercle intérieur rempli (PAPER_COLOR) pour "percer" l'anneau
    """
    outer = plt.Circle((0, 0), r_out, fill=True, color=color, alpha=alpha, linewidth=0, zorder=zorder)
    inner = plt.Circle((0, 0), r_in, fill=True, color=PAPER_COLOR, alpha=1.0, linewidth=0, zorder=zorder + 0.1)
    ax.add_patch(outer)
    ax.add_patch(inner)


# ============================================================
# “ANTI TRAIT GÉANT”
# ============================================================
def max_jump_threshold(R):
    return 0.45 * R


# ============================================================
# TRACÉ CONSTELLATIONS
# ============================================================
def draw_asterisms(ax, aster_segments, clip_patch=None, R=None, linewidth=0.55, alpha=0.60, color="black"):
    jump = max_jump_threshold(R) if R is not None else None

    for seg in aster_segments or []:
        if not seg or len(seg) < 2:
            continue

        p1, p2 = seg[0], seg[1]
        if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 2 and len(p2) == 2):
            continue

        ra1, dec1 = p1[0], p1[1]
        ra2, dec2 = p2[0], p2[1]
        if not (is_valid_coord(ra1, dec1) and is_valid_coord(ra2, dec2)):
            continue

        ra1, dec1 = float(ra1), float(dec1)
        ra2, dec2 = float(ra2), float(dec2)

        ra1, ra2 = unwrap_ra_pair(ra1, ra2)

        x1, y1 = project_star(ra1, dec1)
        x2, y2 = project_star(ra2, dec2)

        if jump is not None and np.hypot(x2 - x1, y2 - y1) > jump:
            continue

        line, = ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=linewidth,
            alpha=alpha,
            color=color,
            linestyle="--",
            zorder=CONSTELLATION_LINE_ZORDER,
        )
        clip_artist(line, clip_patch)


def draw_boundaries(ax, boundary_loops, clip_patch=None, R=None, linewidth=0.45, alpha=0.35, color="black", linestyle="--"):
    jump = max_jump_threshold(R) if R is not None else None

    for loop in boundary_loops or []:
        if not loop or len(loop) < 2:
            continue

        xs, ys = [], []
        prev_ra = None
        prev_x = None
        prev_y = None

        def flush():
            if len(xs) >= 2:
                line, = ax.plot(
                    xs,
                    ys,
                    linewidth=linewidth,
                    alpha=alpha,
                    color=color,
                    linestyle=linestyle,
                    zorder=CONSTELLATION_LINE_ZORDER,
                )
                clip_artist(line, clip_patch)
            xs.clear()
            ys.clear()

        for pt in loop:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                flush()
                prev_ra = None
                prev_x = prev_y = None
                continue

            ra_raw, dec_raw = pt[0], pt[1]
            if not is_valid_coord(ra_raw, dec_raw):
                flush()
                prev_ra = None
                prev_x = prev_y = None
                continue

            ra, dec = float(ra_raw), float(dec_raw)

            if prev_ra is not None:
                prev_ra, ra = unwrap_ra_pair(prev_ra, ra)

            x, y = project_star(ra, dec)

            if jump is not None and prev_x is not None and prev_y is not None:
                if np.hypot(x - prev_x, y - prev_y) > jump:
                    flush()
                    prev_ra = ra
                    prev_x, prev_y = x, y
                    xs.append(x)
                    ys.append(y)
                    continue

            xs.append(x)
            ys.append(y)
            prev_ra = ra
            prev_x, prev_y = x, y

        flush()


def _project_ring(ring):
    xs, ys = [], []
    prev_ra = None
    for pt in ring:
        ra, dec = float(pt[0]), float(pt[1])
        if prev_ra is not None:
            prev_ra, ra = unwrap_ra_pair(prev_ra, ra)
        x, y = project_star(ra, dec)
        xs.append(x)
        ys.append(y)
        prev_ra = ra
    return xs, ys


def _signed_area(xs, ys):
    xa, ya = np.array(xs), np.array(ys)
    return 0.5 * float(np.sum(xa[:-1] * ya[1:] - xa[1:] * ya[:-1]))


def _rings_to_path(rings):
    """Combine plusieurs anneaux GeoJSON en un Path composé pour fill nonzero.

    En projection polaire, certains anneaux ont leur sens de rotation apparent
    inversé par rapport au sens sphérique. On détecte l'anneau extérieur comme
    celui qui a la plus grande aire signée absolue, puis les trous sont forcés
    en sens inverse (CW) pour que la règle nonzero les soustraie correctement.
    """
    projected = []
    for ring in rings:
        xs, ys = _project_ring(ring)
        if len(xs) >= 3:
            projected.append((xs, ys, _signed_area(xs, ys)))

    if not projected:
        return None

    # L'anneau avec la plus grande aire absolue est l'enveloppe extérieure
    projected.sort(key=lambda t: abs(t[2]), reverse=True)

    verts, codes = [], []
    for i, (xs, ys, area) in enumerate(projected):
        if i == 0:
            # Extérieur : doit être CCW (aire > 0) pour nonzero fill
            if area < 0:
                xs, ys = xs[::-1], ys[::-1]
        else:
            # Trous : doivent être CW (aire < 0)
            if area > 0:
                xs, ys = xs[::-1], ys[::-1]
        n = len(xs)
        verts.extend(zip(xs, ys))
        codes += [Path.MOVETO] + [Path.LINETO] * (n - 1)
        codes[-1] = Path.CLOSEPOLY

    return Path(verts, codes)


def draw_milky_way(ax, clip_patch=None):
    if not DRAW_MILKY_WAY:
        return
    try:
        with open(MW_FILE, "r", encoding="utf-8") as f:
            mw = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] {MW_FILE} introuvable – voie lactée ignorée")
        return

    features = mw.get("features", [])
    n = len(features)
    if n == 0:
        return

    for i, feature in enumerate(features):
        t = i / max(n - 1, 1)
        alpha = MW_ALPHA_MIN + t * (MW_ALPHA_MAX - MW_ALPHA_MIN)

        geom = feature.get("geometry", {})
        gtype = geom.get("type", "")
        polygons = geom.get("coordinates", [])
        if gtype == "Polygon":
            polygons = [polygons]
        elif gtype != "MultiPolygon":
            continue

        for poly in polygons:
            # poly = [exterior_ring, hole1, hole2, ...]
            # On combine en un seul Path : le fill nonzero coupe les trous (CW)
            path = _rings_to_path(poly)
            if path is None:
                continue
            patch = PathPatch(path, facecolor=MW_COLOR, edgecolor="none",
                              alpha=alpha, linewidth=0, zorder=-5)
            ax.add_patch(patch)
            clip_artist(patch, clip_patch)


def offset_radially(x, y, delta):
    r = np.hypot(x, y)
    if r == 0:
        return x, y
    ux, uy = x / r, y / r
    return x + ux * delta, y + uy * delta


def draw_constellation_label(
    ax,
    centrum,
    text,
    clip_patch=None,
    R=None,
    fontsize=5,
    alpha=1.0,
    occupied_bboxes=None,
    renderer=None,
    collision_pad_px=0.0,
    extra_bboxes=None,
):
    """
    Place un label de constellation au plus proche possible du centrum, sans chevauchement.
    - la liste occupied_bboxes est partagée avec les étoiles (points + labels).
    """
    if not centrum or "ra" not in centrum or "dec" not in centrum:
        return None
    if not is_valid_coord(centrum["ra"], centrum["dec"]):
        return None

    if renderer is None:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()

    x0, y0 = project_star(float(centrum["ra"]), float(centrum["dec"]))

    radial_offsets = [0.0, 1.2, -1.2, 2.5, -2.5, 4.0, -4.0, 5.5, -5.5]
    tangential_offsets = [0.0, -1.5, 1.5, -3.0, 3.0, -4.5, 4.5]

    best = None
    best_d2 = None
    best_bbox = None

    for radial_delta in radial_offsets:
        rx, ry = offset_radially(x0, y0, radial_delta)
        for tang_delta in tangential_offsets:
            tx, ty = offset_tangential(rx, ry, tang_delta)

            if R is not None and not is_inside_disk_xy(tx, ty, R):
                continue

            rot = get_tangent_rotation(tx, ty)
            t = ax.text(
                tx,
                ty,
                text.upper(),
                fontsize=fontsize,
                rotation=rot,
                rotation_mode="anchor",
                ha="center",
                va="center",
                alpha=alpha,
                color=CONST_LABEL_COLOR,
                fontproperties=gilroy_medium,
                zorder=60,
            )
            t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
            clip_artist(t, clip_patch)

            ok = register_text_if_no_overlap(
                ax,
                t,
                occupied_bboxes=occupied_bboxes,
                renderer=renderer,
                pad_px=collision_pad_px,
                extra_bboxes=extra_bboxes,
                register_bbox=False,
            )
            if not ok:
                continue

            d2 = (tx - x0) ** 2 + (ty - y0) ** 2
            bbox = padded_bbox(t.get_window_extent(renderer=renderer), pad_px=collision_pad_px)

            if best is None or d2 < best_d2:
                if best is not None:
                    best.remove()
                best = t
                best_d2 = d2
                best_bbox = bbox
            else:
                t.remove()

    if best is not None and occupied_bboxes is not None and best_bbox is not None:
        occupied_bboxes.append(best_bbox)

    if best is not None and DRAW_LEADER_LINES and best_d2 > LEADER_CONST_MIN_DU ** 2:
        tx_b, ty_b = best.get_position()
        leader, = ax.plot([x0, tx_b], [y0, ty_b],
                          lw=LEADER_LINE_LW, color=CONST_LABEL_COLOR,
                          alpha=LEADER_LINE_ALPHA, zorder=55,
                          solid_capstyle="round")
        clip_artist(leader, clip_patch)

    return best


# ============================================================
# HORIZON (points de la fenêtre)
# ============================================================
H_vals = np.linspace(0, 2 * np.pi, 1000)
delta_hor = np.degrees(np.arctan(-np.cos(H_vals) / np.tan(phi)))

r_hor = 90 - delta_hor
x_hor = r_hor * np.sin(H_vals)
y_hor = -r_hor * np.cos(H_vals)

max_radius = float(np.max(r_hor))
outer_radius = max_radius * OUTER_RING_FACTOR


# ============================================================
# PAGE 2 : polygone fenêtre (horizon + arc du cercle)
# ============================================================
def arc_points(radius, a_start, a_end, n=600):
    aa = np.linspace(a_start, a_end, n)
    return radius * np.cos(aa), radius * np.sin(aa)


def polygon_area(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def build_horizon_window_polygon(hx, hy, radius):
    hx = np.asarray(hx)
    hy = np.asarray(hy)

    p0 = np.array([hx[0], hy[0]])
    p1 = np.array([hx[-1], hy[-1]])

    a0 = np.arctan2(p0[1], p0[0])
    a1 = np.arctan2(p1[1], p1[0])

    ax_d1, ay_d1 = arc_points(radius, a0, a1, n=700)  # a0 -> a1
    ax_d2, ay_d2 = arc_points(radius, a1, a0, n=700)  # a1 -> a0

    x1 = np.concatenate([hx, ax_d2, [hx[0]]])
    y1 = np.concatenate([hy, ay_d2, [hy[0]]])

    x2 = np.concatenate([hx, ax_d1, [hx[0]]])
    y2 = np.concatenate([hy, ay_d1, [hy[0]]])

    area1 = abs(polygon_area(x1, y1))
    area2 = abs(polygon_area(x2, y2))

    disk_area = np.pi * radius * radius

    def score(area):
        ratio = area / disk_area
        if ratio < 0.01 or ratio > 0.99:
            return 1e9
        return abs(ratio - 0.35)

    if score(area1) <= score(area2):
        return x1, y1
    return x2, y2


# ============================================================
# Heures : 00h haut, 06h gauche, 12h bas, 18h droite
# ============================================================
def hour_to_xy(R, hour_float):
    ang = -2 * np.pi * (hour_float / 24.0)  # signe - => 6h gauche / 18h droite
    x = R * np.sin(ang)
    y = R * np.cos(ang)
    return x, y


def draw_hour_ring(ax, R_text, R_tick_out, R_tick_in_15, R_tick_in_30, R_tick_in_60):
    for k in range(0, 24 * 4):
        h = k / 4.0
        x0, y0 = hour_to_xy(R_tick_out, h)
        if k % 4 == 0:
            Rin = R_tick_in_60
            lw = 1.0
        elif k % 2 == 0:
            Rin = R_tick_in_30
            lw = 0.9
        else:
            Rin = R_tick_in_15
            lw = 0.7
        x1, y1 = hour_to_xy(Rin, h)
        ax.plot([x0, x1], [y0, y1], linewidth=lw, color="black", alpha=0.95)

    for k in range(0, 24 * 2):
        h = k / 2.0
        x, y = hour_to_xy(R_text, h)

        hh = int(h) % 24
        mm = 30 if (k % 2 == 1) else 0
        label = f"{hh:02d}h" if mm == 0 else f"{hh:02d}h{mm:02d}"

        rot = get_tangent_rotation(x, y) + 180
        ax.text(
            x, y, label,
            fontsize=HOUR_LABEL_FONT,
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color="black",
            fontproperties=dmmono,
        )


# ============================================================
# Cardinaux sur la courbe d'horizon, tangents à la courbe
# ============================================================
def pick_horizon_point_near_direction(hx, hy, target_vec):
    hx = np.asarray(hx)
    hy = np.asarray(hy)
    v = np.column_stack([hx, hy])
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    v_unit = v / norms[:, None]

    t = np.array(target_vec, dtype=float)
    t_norm = np.linalg.norm(t) or 1.0
    t = t / t_norm

    dots = v_unit @ t
    i = int(np.argmax(dots))
    return float(hx[i]), float(hy[i]), i


def get_polyline_tangent_rotation(px, py, idx):
    px = np.asarray(px)
    py = np.asarray(py)
    n = len(px)
    if n < 2:
        return 0.0

    i0 = max(0, idx - 1)
    i1 = min(n - 1, idx + 1)
    if i0 == i1:
        if idx == 0:
            i1 = 1
        else:
            i0 = n - 2

    dx = float(px[i1] - px[i0])
    dy = float(py[i1] - py[i0])
    angle = np.degrees(np.arctan2(dy, dx))

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def get_zenith_xy(latitude_deg):
    r_zen = 90.0 - float(latitude_deg)
    return 0.0, -r_zen


def draw_text_on_arc(ax, text, radius, theta_center_deg, char_step_deg, fontsize, color, fontproperties, zorder=40):
    if not text:
        return

    n = len(text)
    if n == 1:
        thetas = [theta_center_deg]
    else:
        half_span = 0.5 * (n - 1) * char_step_deg
        thetas = np.linspace(theta_center_deg + half_span, theta_center_deg - half_span, n)

    for ch, th_deg in zip(text, thetas):
        if ch == " ":
            continue
        th = np.radians(th_deg)
        x = radius * np.cos(th)
        y = radius * np.sin(th)

        rot = th_deg - 90.0
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180

        ax.text(
            x, y, ch,
            fontsize=fontsize,
            ha="center",
            va="center",
            color=color,
            rotation=rot,
            rotation_mode="anchor",
            fontproperties=fontproperties,
            zorder=zorder,
        )


def draw_horizon_top_text(ax, text, radius, padding):
    if not text:
        return

    text_radius = max(0.0, radius - padding)
    draw_text_on_arc(
        ax=ax,
        text=text,
        radius=text_radius,
        theta_center_deg=90.0,
        char_step_deg=HORIZON_TOP_TEXT_CHAR_STEP_DEG,
        fontsize=HORIZON_TOP_TEXT_FONT,
        color="white",
        fontproperties=dmmono,
        zorder=40,
    )

    draw_text_on_arc(
        ax=ax,
        text="Carte générée avec Astroshare (https://astroshare.fr)",
        radius=max(0.0, radius - padding) - 5,
        theta_center_deg=90.0,
        char_step_deg=HORIZON_TOP_TEXT_CHAR_STEP_DEG,
        fontsize=HORIZON_TOP_TEXT_FONT,
        color="white",
        fontproperties=dmmono,
        zorder=40,
    )


def draw_cardinals_on_horizon(ax, hx, hy, delta_out):
    card = [
        ("Nord", (0, 1)),
        ("Est", (-1, 0)),
        ("Sud", (0, -1)),
        ("Ouest", (1, 0)),
    ]

    for label, vec in card:
        x, y, i = pick_horizon_point_near_direction(hx, hy, vec)
        tx, ty = offset_radially(x, y, delta_out)

        rot = get_polyline_tangent_rotation(hx, hy, i) + 180
        ax.text(
            tx, ty, label,
            fontsize=CARDINAL_FONT,
            ha="center",
            va="center",
            color="black" if label == "Sud" else "white",
            rotation=rot,
            rotation_mode="anchor",
            fontproperties=gilroy_black,
        )


# ============================================================
# Dates (mois + jours) : placement sur bords des anneaux
# ============================================================
months = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def day_of_year(month_index_0, day):
    return sum(MONTH_DAYS[:month_index_0]) + day


REF_DOY = day_of_year(2, 21)      # 21 mars
SIDEREAL_YEAR = 365.2422


def sun_ra_hours_from_doy(doy):
    delta = (doy - REF_DOY) / SIDEREAL_YEAR
    return (24.0 * delta) % 24.0


def place_on_ring_from_ra(R, ra_hours):
    H = -np.radians(ra_hours * 15.0)
    x = R * np.sin(H)
    y = -R * np.cos(H)
    return x, y


def draw_cut_circle(ax, radius):
    c = plt.Circle(
        (0, 0),
        radius,
        fill=False,
        linewidth=CUT_CIRCLE_LW,
        linestyle=CUT_CIRCLE_STYLE,
        color=CUT_CIRCLE_COLOR,
        zorder=1000,
    )
    ax.add_patch(c)


# ============================================================
# EXPORT PDF
# ============================================================
OUTPUT_PATH = "output/starmap.pdf"
os.makedirs("output", exist_ok=True)
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

with PdfPages(OUTPUT_PATH) as pdf:
    # ======================
    # PAGE 1 : CARTE DU CIEL + ZONE ROSE (MOIS + JOURS)
    # ======================
    fig = plt.figure(figsize=FIGURE_SIZE_A4_IN)
    ax = setup_map_ax(fig, outer_radius)

    R_pink_in = max_radius * PINK_IN
    R_pink_out = max_radius * PINK_OUT

    if DRAW_PINK_BAND:
        add_annulus(
            ax,
            r_in=R_pink_in,
            r_out=R_pink_out,
            color=PINK_COLOR,
            alpha=BAND_ALPHA,
            zorder=-10,
        )

    border_circle = plt.Circle((0, 0), max_radius, fill=False, linewidth=1, color="black")
    ax.add_patch(border_circle)

    clip_circle = plt.Circle((0, 0), max_radius, transform=ax.transData) if CLIP_SKY else None

    draw_milky_way(ax, clip_patch=clip_circle)

    fig.canvas.draw()
    label_renderer = fig.canvas.get_renderer()

    # Obstacles = DOTS (points d'étoiles) uniquement
    star_dot_bboxes = []
    for star in visible_stars:
        x_star, y_star = project_star(star["ra"], star["dec"])
        if not is_inside_disk_xy(x_star, y_star, max_radius):
            continue
        _, marker_area_star, _ = star_marker_style(star)
        star_dot_bboxes.append(
            marker_bbox_from_data(
                ax,
                x_star,
                y_star,
                marker_area_star,
                pad_px=STAR_DOT_COLLISION_PADDING_PX,
            )
        )


    # Obstacles = TEXTES déjà placés (étoiles + constellations)
    label_bboxes = []

    # Horizon (optionnel)
    # horizon_line, = ax.plot(x_hor, y_hor, linewidth=1, color="black", alpha=1.0)
    # clip_artist(horizon_line, clip_circle)

    # Constellations : lignes d'abord, labels après les étoiles (priorité aux étoiles)
    constellation_labels = []
    if constellations:
        for c in constellations:
            aster = c.get("aster", []) or []
            boundaries = c.get("boundaries", c.get("boundary", [])) or []
            centrum = c.get("centrum", None)

            if aster:
                draw_asterisms(ax, aster, clip_patch=clip_circle, R=max_radius, linewidth=0.55, alpha=0.65, color=CONST_LINE_COLOR)

            if DRAW_CONSTELLATION_BOUNDARIES and boundaries:
                draw_boundaries(ax, boundaries, clip_patch=clip_circle, R=max_radius, linewidth=0.45, alpha=0.30, color=CONST_LINE_COLOR, linestyle="--")

            if DRAW_CONSTELLATION_LABELS and c.get("name") and centrum:
                constellation_labels.append((centrum, c["name"]))

    # Étoiles + labels (priorité)
    for star in visible_stars:
        x, y = project_star(star["ra"], star["dec"])
        name, marker_area, marker_color = star_marker_style(star)

        sc = ax.scatter(x, y, s=marker_area, color=marker_color, zorder=STAR_DOT_ZORDER)
        clip_artist(sc, clip_circle)

        # Labels d'étoiles (priorité) : anti-chevauchement, au plus proche possible du point de base.
        # Note: Polaris (ids "Lodestar") est toujours labellisée, même si sa magnitude dépasse le seuil.
        if name and (name == "Lodestar" or star["V"] <= LABEL_MAG_LIMIT):
            display_name = "Polaris" if name == "Lodestar" else name

            forced_star = (display_name in FORCED_STAR_NAMES)

            place_star_label(
                ax=ax,
                x=x,
                y=y,
                label=display_name,
                marker_area_points2=marker_area,
                occupied_bboxes=label_bboxes,
                renderer=label_renderer,
                extra_bboxes=None if forced_star else star_dot_bboxes,
                clip_patch=clip_circle,
                disk_R=max_radius - LABEL_DISK_MARGIN,
                fontsize=6,
                collision_pad_px=LABEL_COLLISION_PADDING_PX,
                color=marker_color,
            )

    # Labels de constellations (après les étoiles => priorité aux labels d'étoiles)
    if constellation_labels:
        for centrum, cname in constellation_labels:
            forced_const = (cname in FORCED_CONSTELLATION_NAMES)
            draw_constellation_label(
                ax,
                centrum,
                cname,
                clip_patch=clip_circle,
                R=max_radius - LABEL_DISK_MARGIN,
                fontsize=6,
                alpha=1,
                occupied_bboxes=label_bboxes,
                renderer=label_renderer,
                collision_pad_px=LABEL_COLLISION_PADDING_PX,
                extra_bboxes=None if (forced_const or not CONST_LABEL_AVOID_STAR_DOTS) else star_dot_bboxes,
            )

    # ------------------------------------------------------------
    # MOIS + JOURS : placement sur les lignes (bords) de la zone rose
    # ------------------------------------------------------------
    R_months = R_pink_in + TEXT_PAD_IN
    R_days = R_pink_out - TEXT_PAD_OUT

    R_day_marker = (R_pink_out - DAY_TICK_OUT_PAD) - DAY_TICK_LEN

    R_month_sep_out = R_pink_out - MONTH_SEPARATOR_OUT_PAD
    R_month_sep_in = R_pink_in + MONTH_SEPARATOR_IN_PAD

    days_to_show = [5, 10, 15, 20, 25]
    dates_clip_patch = clip_circle if (CLIP_SKY and CLIP_DATES) else None

    if DRAW_MONTH_SEPARATORS:
        for m_idx in range(12):
            doy_sep = day_of_year(m_idx, 1)
            ra_sep = sun_ra_hours_from_doy(doy_sep)
            xs0, ys0 = place_on_ring_from_ra(R_month_sep_out, ra_sep)
            xs1, ys1 = place_on_ring_from_ra(R_month_sep_in, ra_sep)
            sep, = ax.plot(
                [xs0, xs1], [ys0, ys1],
                linewidth=MONTH_SEPARATOR_LW,
                color=MONTH_SEPARATOR_COLOR,
                alpha=1.0,
                zorder=20,
            )
            clip_artist(sep, dates_clip_patch)

    for m_idx, m_name in enumerate(months):
        doy_m = day_of_year(m_idx, 15)
        ra_m = sun_ra_hours_from_doy(doy_m)
        x_m, y_m = place_on_ring_from_ra(R_months, ra_m)

        rot_m = get_tangent_rotation(x_m, y_m) + 180
        t = ax.text(
            x_m, y_m, m_name.upper(),
            fontsize=10,
            rotation=rot_m,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color="white",
            fontproperties=gilroy_black,
        )
        clip_artist(t, dates_clip_patch)

        for d in days_to_show:
            if d > MONTH_DAYS[m_idx]:
                continue

            doy_d = day_of_year(m_idx, d)
            ra_d = sun_ra_hours_from_doy(doy_d)

            xm, ym = place_on_ring_from_ra(R_day_marker, ra_d)
            day_marker = plt.Circle(
                (xm, ym),
                DAY_MARKER_RADIUS,
                fill=True,
                color="white",
                linewidth=0,
                zorder=30,
            )
            ax.add_patch(day_marker)
            clip_artist(day_marker, dates_clip_patch)

            x_d, y_d = place_on_ring_from_ra(R_days, ra_d)
            rot_d = get_tangent_rotation(x_d, y_d) + 180
            t = ax.text(
                x_d, y_d, str(d),
                fontsize=7,
                rotation=rot_d,
                rotation_mode="anchor",
                ha="center",
                va="center",
                color="white",
                fontproperties=dmmono,
            )
            clip_artist(t, dates_clip_patch)

    if DRAW_CUT_CIRCLE:
        draw_cut_circle(ax, max_radius * CUT_RADIUS_FACTOR)

    fig.text(
        0.5, TITLE_Y,
        f"Carte du ciel - Latitude {LATITUDE}°",
        ha="center",
        fontsize=14,
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ======================
    # PAGE 2 : MASQUE + ZONE BLEUE (HEURES) + CARDINAUX + AXE N-S
    # ======================
    fig2 = plt.figure(figsize=FIGURE_SIZE_A4_IN)
    ax2 = setup_map_ax(fig2, outer_radius)

    R_blue_in = max_radius * BLUE_IN
    R_blue_out = max_radius * BLUE_OUT

    if DRAW_BLUE_BAND:
        add_annulus(
            ax2,
            r_in=R_blue_in,
            r_out=R_blue_out,
            color=BLUE_COLOR,
            alpha=BAND_ALPHA,
            zorder=-10,
        )

    disk = plt.Circle((0, 0), max_radius, fill=True, color=MASK_COLOR, linewidth=0)
    ax2.add_patch(disk)

    if DRAW_PAGE2_CENTER_CROSS:
        s = PAGE2_CENTER_CROSS_HALF_SIZE
        ax2.plot(
            [-s, s], [0, 0],
            color=PAGE2_CENTER_CROSS_COLOR,
            linewidth=PAGE2_CENTER_CROSS_LW,
            zorder=30,
            solid_capstyle="round",
        )
        ax2.plot(
            [0, 0], [-s, s],
            color=PAGE2_CENTER_CROSS_COLOR,
            linewidth=PAGE2_CENTER_CROSS_LW,
            zorder=30,
            solid_capstyle="round",
        )

    hx = np.asarray(x_hor)
    hy = np.asarray(y_hor)
    wx, wy = build_horizon_window_polygon(hx, hy, max_radius)
    ax2.fill(wx, wy, color=PAPER_COLOR, linewidth=0)
    window_poly = Polygon(np.column_stack([wx, wy]), closed=True, facecolor="none", edgecolor="none")
    ax2.add_patch(window_poly)

    if DRAW_HORIZON_TOP_TEXT:
        draw_horizon_top_text(ax2, HORIZON_TOP_TEXT, max_radius, HORIZON_TOP_TEXT_PADDING)

    if DRAW_ZENITH_MARKER:
        zx, zy = get_zenith_xy(LATITUDE)
        if is_inside_disk_xy(zx, zy, max_radius):
            zenith_marker = plt.Circle(
                (zx, zy),
                ZENITH_MARKER_RADIUS,
                fill=True,
                color=ZENITH_MARKER_COLOR,
                linewidth=0,
                zorder=25,
            )
            zenith_marker.set_clip_path(window_poly)
            ax2.add_patch(zenith_marker)

    if DRAW_VERTICAL_WINDOW_LINE:
        ns_line, = ax2.plot([0, 0], [max_radius, -max_radius], color="black", linewidth=WINDOW_LINE_WIDTH)
        ns_line.set_clip_path(window_poly)

    if DRAW_CARDINALS_ON_HORIZON:
        draw_cardinals_on_horizon(ax2, hx, hy, CARDINAL_OFFSET)

    if DRAW_HORIZON_OUTLINE_ON_MASK:
        ax2.plot(hx, hy, linewidth=1, color="black")

    R_hours_text = R_blue_out - TEXT_PAD_OUT

    R_hour_tick_base = R_blue_in + 0.2
    R_hour_tick_15 = R_hour_tick_base + 2.0
    R_hour_tick_30 = R_hour_tick_base + 3.0
    R_hour_tick_60 = R_hour_tick_base + 4.2

    R_hour_tick_15 = min(R_hour_tick_15, R_blue_out - 0.2)
    R_hour_tick_30 = min(R_hour_tick_30, R_blue_out - 0.2)
    R_hour_tick_60 = min(R_hour_tick_60, R_blue_out - 0.2)

    draw_hour_ring(ax2, R_hours_text, R_hour_tick_base, R_hour_tick_15, R_hour_tick_30, R_hour_tick_60)

    if DRAW_CUT_CIRCLE:
        draw_cut_circle(ax2, max_radius * CUT_RADIUS_FACTOR)

    fig2.text(
        0.5, TITLE_Y,
        "Masque d'horizon + anneau des heures",
        ha="center",
        fontsize=14,
    )

    # ------------------------------------------------------------
    # Logo page 2
    # ------------------------------------------------------------
    if os.path.exists(LOGO_PATH):
        logo = mpimg.imread(LOGO_PATH)

        h, w = logo.shape[0], logo.shape[1]
        ratio = h / w

        width = LOGO_SIZE
        height = LOGO_SIZE * ratio

        ax_logo = fig2.add_axes([LOGO_X - width/2, LOGO_Y, width, height])
        ax_logo.imshow(logo, interpolation="none")
        ax_logo.axis("off")

    else:
        print(f"Logo not found at path: {LOGO_PATH}")

    pdf.savefig(fig2)
    plt.close(fig2)
