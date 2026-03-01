import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
from matplotlib import font_manager as fm

# ============================================================
# PARAMÈTRES
# ============================================================
LATITUDE = 45
MAX_MAGNITUDE = 3
LABEL_MAG_LIMIT = 1  # seuil pour afficher le nom des étoiles

# Constellations
CONSTELLATIONS_FILE = "constellations_fr.json"
DRAW_CONSTELLATION_BOUNDARIES = False
DRAW_CONSTELLATION_LABELS = True

# Clipping
CLIP_SKY = True
CLIP_DATES = False  # si True, mois/jours clippés au disque

# Page 2 (masque simple)
MASK_COLOR = "#000000"  # disque plein
PAPER_COLOR = "white"   # couleur simulant la transparence (papier blanc)
DRAW_HORIZON_OUTLINE_ON_MASK = True

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

# ============================================================
# POLICE PERSONNALISÉE
# ============================================================
GILROY_BLACK_FONT_PATH = "fonts/Gilroy-Black.ttf"
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

# ------------------------------------------------------------
# Mise en page A4 / marges d'impression
# ------------------------------------------------------------
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
CM_PER_INCH = 2.54
PRINT_MARGIN_CM = 1.5

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

        line, = ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha, color=color, linestyle="--")
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
                line, = ax.plot(xs, ys, linewidth=linewidth, alpha=alpha, color=color, linestyle=linestyle)
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


def draw_constellation_label(ax, centrum, text, clip_patch=None, R=None, fontsize=5, alpha=1.0):
    if not centrum or "ra" not in centrum or "dec" not in centrum:
        return
    if not is_valid_coord(centrum["ra"], centrum["dec"]):
        return

    x, y = project_star(float(centrum["ra"]), float(centrum["dec"]))

    if R is not None and not is_inside_disk_xy(x, y, R):
        return

    rot = get_tangent_rotation(x, y)
    t = ax.text(x,
                y,
                text.upper(),
                fontsize=fontsize,
                rotation=rot,
                rotation_mode="anchor",
                ha="center",
                va="center",
                alpha=alpha,
                color="black",
                )
    clip_artist(t, clip_patch)


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

    # fenêtre = horizon (p0->p1) + arc (p1->p0)
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
    # ticks toutes les 15 min (96 ticks)
    for k in range(0, 24 * 4):
        h = k / 4.0
        x0, y0 = hour_to_xy(R_tick_out, h)   # base (bord interne)
        if k % 4 == 0:        # heure pleine
            Rin = R_tick_in_60
            lw = 1.0
        elif k % 2 == 0:      # demi-heure
            Rin = R_tick_in_30
            lw = 0.9
        else:                 # quart d'heure
            Rin = R_tick_in_15
            lw = 0.7
        x1, y1 = hour_to_xy(Rin, h)          # pointe (vers l'extérieur)
        ax.plot([x0, x1], [y0, y1], linewidth=lw, color="black", alpha=0.95)

    # labels toutes les 30 min (48 labels)
    for k in range(0, 24 * 2):
        h = k / 2.0
        x, y = hour_to_xy(R_text, h)

        hh = int(h) % 24
        mm = 30 if (k % 2 == 1) else 0
        label = f"{hh:02d}h" if mm == 0 else f"{hh:02d}h{mm:02d}"

        # Lisible "au-dessus" (côté extérieur) comme sur ton exemple :
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


def offset_radially(x, y, delta):
    r = np.hypot(x, y)
    if r == 0:
        return x, y
    ux, uy = x / r, y / r
    return x + ux * delta, y + uy * delta


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

    # Garde le texte lisible (évite les rotations tête en bas)
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def get_zenith_xy(latitude_deg):
    # Dans ce repère local, le zénith est à H=0 et dec=latitude.
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
    # Convention demandée :
    # N en haut, E à gauche, S en bas, O à droite
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
        zorder=1000,  # au-dessus de tout
    )
    ax.add_patch(c)


# ============================================================
# EXPORT PDF
# ============================================================
os.makedirs("output", exist_ok=True)

with PdfPages("output/starmap.pdf") as pdf:
    # ======================
    # PAGE 1 : CARTE DU CIEL + ZONE ROSE (MOIS + JOURS)
    # ======================
    fig = plt.figure(figsize=FIGURE_SIZE_A4_IN)  # A4
    ax = fig.add_axes([MARGIN_LEFT, MARGIN_BOTTOM, MARGIN_WIDTH, MARGIN_HEIGHT])

    # Rayons des bords de la zone rose
    R_pink_in = max_radius * PINK_IN
    R_pink_out = max_radius * PINK_OUT

    # Fond zone rose
    if DRAW_PINK_BAND:
        add_annulus(
            ax,
            r_in=R_pink_in,
            r_out=R_pink_out,
            color=PINK_COLOR,
            alpha=BAND_ALPHA,
            zorder=-10,
        )

    # Bordure du disque principal
    border_circle = plt.Circle((0, 0), max_radius, fill=False, linewidth=1, color="black")
    ax.add_patch(border_circle)

    # Cercle de clipping (invisible) : uniquement le disque principal
    clip_circle = plt.Circle((0, 0), max_radius, transform=ax.transData) if CLIP_SKY else None

    # Horizon
    # horizon_line, = ax.plot(x_hor, y_hor, linewidth=1, color="black", alpha=1.0)
    # clip_artist(horizon_line, clip_circle)

    # Constellations
    if constellations:
        for c in constellations:
            aster = c.get("aster", []) or []
            boundaries = c.get("boundaries", c.get("boundary", [])) or []
            centrum = c.get("centrum", None)

            if aster:
                draw_asterisms(ax, aster, clip_patch=clip_circle, R=max_radius, linewidth=0.55, alpha=0.60, color="black")

            if DRAW_CONSTELLATION_BOUNDARIES and boundaries:
                draw_boundaries(ax, boundaries, clip_patch=clip_circle, R=max_radius, linewidth=0.45, alpha=0.30, color="black", linestyle="--")

            if DRAW_CONSTELLATION_LABELS and c.get("name"):
                draw_constellation_label(ax, centrum, c["name"], clip_patch=clip_circle, R=max_radius, fontsize=5, alpha=1)

    # Étoiles
    for star in visible_stars:
        x, y = project_star(star["ra"], star["dec"])
        size = max(1, 4 - star["V"])

        if extract_star_name(star.get("ids")) == "Lodestar":
            sc = ax.scatter(x, y, s=size**4, color="red")
            clip_artist(sc, clip_circle)

            t = ax.text(
                x + 3, y + 3, "Polaris",
                fontsize=5,
                rotation=get_tangent_rotation(x, y),
                rotation_mode="anchor",
                ha="center",
                va="center",
            )
            clip_artist(t, clip_circle)
        else:
            sc = ax.scatter(x, y, s=size**2, color="black")
            clip_artist(sc, clip_circle)

        if star["V"] <= LABEL_MAG_LIMIT:
            name = extract_star_name(star.get("ids"))
            if name:
                tx, ty = x * 1.07, y * 1.07
                if is_inside_disk_xy(tx, ty, max_radius):
                    rotation = get_tangent_rotation(x, y)
                    t = ax.text(
                        tx, ty, name,
                        fontsize=5,
                        rotation=rotation,
                        rotation_mode="anchor",
                        ha="center",
                        va="center",
                    )
                    clip_artist(t, clip_circle)

    # ------------------------------------------------------------
    # MOIS + JOURS : placement sur les lignes (bords) de la zone rose
    # -> Mois : sur la ligne intérieure (R_pink_in + pad)
    # -> Jours : sur la ligne extérieure (R_pink_out - pad), orientés vers l'extérieur (+180)
    # ------------------------------------------------------------
    R_months = R_pink_in + TEXT_PAD_IN
    R_days = R_pink_out - TEXT_PAD_OUT

    # Marqueur jour : positionné radialement sous la date
    R_day_marker = (R_pink_out - DAY_TICK_OUT_PAD) - DAY_TICK_LEN

    # Séparateurs mois : petits traits radiaux entre les mois
    R_month_sep_out = R_pink_out - MONTH_SEPARATOR_OUT_PAD
    R_month_sep_in = R_pink_in + MONTH_SEPARATOR_IN_PAD

    days_to_show = [5, 10, 15, 20, 25]
    dates_clip_patch = clip_circle if (CLIP_SKY and CLIP_DATES) else None

    if DRAW_MONTH_SEPARATORS:
        for m_idx in range(12):
            doy_sep = day_of_year(m_idx, 1)  # début de mois
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
        # mois : RA du Soleil au 15 du mois
        doy_m = day_of_year(m_idx, 15)
        ra_m = sun_ra_hours_from_doy(doy_m)
        x_m, y_m = place_on_ring_from_ra(R_months, ra_m)

        # mois : lisible côté extérieur (comme un planisphère)
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

        # jours (marqueur + texte)
        for d in days_to_show:
            if d > MONTH_DAYS[m_idx]:
                continue

            doy_d = day_of_year(m_idx, d)
            ra_d = sun_ra_hours_from_doy(doy_d)

            # petit cercle blanc sous la date
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

            # texte jour
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

    # Cercle de découpe (au-dessus de tout)
    if DRAW_CUT_CIRCLE:
        cut_r = max_radius * CUT_RADIUS_FACTOR
        draw_cut_circle(ax, cut_r)

    # Mise en page Page 1
    ax.set_xlim(-outer_radius, outer_radius)
    ax.set_ylim(-outer_radius, outer_radius)
    ax.set_aspect("equal")
    ax.axis("off")

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
    fig2 = plt.figure(figsize=FIGURE_SIZE_A4_IN)  # A4
    ax2 = fig2.add_axes([MARGIN_LEFT, MARGIN_BOTTOM, MARGIN_WIDTH, MARGIN_HEIGHT])

    # Rayons des bords de la zone bleue
    R_blue_in = max_radius * BLUE_IN
    R_blue_out = max_radius * BLUE_OUT

    # Fond zone bleue
    if DRAW_BLUE_BAND:
        add_annulus(
            ax2,
            r_in=R_blue_in,
            r_out=R_blue_out,
            color=BLUE_COLOR,
            alpha=BAND_ALPHA,
            zorder=-10,
        )

    # Disque plein
    disk = plt.Circle((0, 0), max_radius, fill=True, color=MASK_COLOR, linewidth=0)
    ax2.add_patch(disk)

    # Fenêtre d'horizon "blanche" (simulation transparence)
    hx = np.asarray(x_hor)
    hy = np.asarray(y_hor)
    wx, wy = build_horizon_window_polygon(hx, hy, max_radius)
    ax2.fill(wx, wy, color=PAPER_COLOR, linewidth=0)
    window_poly = Polygon(np.column_stack([wx, wy]), closed=True, facecolor="none", edgecolor="none")
    ax2.add_patch(window_poly)

    # Texte courbé en haut du masque, sur le bord interne du disque
    if DRAW_HORIZON_TOP_TEXT:
        draw_horizon_top_text(ax2, HORIZON_TOP_TEXT, max_radius, HORIZON_TOP_TEXT_PADDING)

    # Marqueur du zénith dans la fenêtre d'horizon
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

    # Axe N-S à travers la fenêtre (clippé par la fenêtre)
    if DRAW_VERTICAL_WINDOW_LINE:
        ns_line, = ax2.plot([0, 0], [max_radius, -max_radius], color="black", linewidth=WINDOW_LINE_WIDTH)
        ns_line.set_clip_path(window_poly)

    # Cardinaux sur le bord intérieur (courbe horizon), tangents à la courbe
    if DRAW_CARDINALS_ON_HORIZON:
        draw_cardinals_on_horizon(ax2, hx, hy, CARDINAL_OFFSET)

    # Contour horizon
    if DRAW_HORIZON_OUTLINE_ON_MASK:
        ax2.plot(hx, hy, linewidth=1, color="black")

    # ------------------------------------------------------------
    # HEURES : posées sur la ligne extérieure de la zone bleue
    # -> Texte sur R_blue_out - pad
    # -> Ticks du bord extérieur vers l'intérieur
    # ------------------------------------------------------------
    R_hours_text = R_blue_out - TEXT_PAD_OUT

    # Ticks "accrochés" à la bordure INTERNE, et qui montent vers l'extérieur (dans le bleu)
    R_hour_tick_base = R_blue_in + 0.2   # point de départ des ticks (dans le bleu, juste au-dessus de la bordure)
    R_hour_tick_15 = R_hour_tick_base + 2.0
    R_hour_tick_30 = R_hour_tick_base + 3.0
    R_hour_tick_60 = R_hour_tick_base + 4.2

    # Sécurité : ne jamais dépasser la bordure extérieure du bleu
    R_hour_tick_15 = min(R_hour_tick_15, R_blue_out - 0.2)
    R_hour_tick_30 = min(R_hour_tick_30, R_blue_out - 0.2)
    R_hour_tick_60 = min(R_hour_tick_60, R_blue_out - 0.2)

    draw_hour_ring(ax2, R_hours_text, R_hour_tick_base, R_hour_tick_15, R_hour_tick_30, R_hour_tick_60)

    # Cercle de découpe (au-dessus de tout)
    if DRAW_CUT_CIRCLE:
        cut_r = max_radius * CUT_RADIUS_FACTOR
        draw_cut_circle(ax2, cut_r)

    # Mise en page Page 2
    ax2.set_xlim(-outer_radius, outer_radius)
    ax2.set_ylim(-outer_radius, outer_radius)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig2.text(
        0.5, TITLE_Y,
        "Masque d'horizon + anneau des heures",
        ha="center",
        fontsize=14,
    )

    pdf.savefig(fig2)
    plt.close(fig2)
