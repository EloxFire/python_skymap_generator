import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

LATITUDE = 45
MAX_MAGNITUDE = 3
LABEL_MAG_LIMIT = 1  # seuil pour afficher le nom

# Constellations
CONSTELLATIONS_FILE = "constellations.json"  # <-- adapte si besoin
DRAW_CONSTELLATION_BOUNDARIES = True
DRAW_CONSTELLATION_LABELS = True

phi = np.radians(LATITUDE)

# -----------------------
# Charger JSON étoiles
# -----------------------
with open('stars.json', 'r') as f:
    data = json.load(f)

stars = data['data']

# -----------------------
# Charger JSON constellations
# Format attendu:
# { "data": [ { "abbr": "...", "name": "...", "aster": [ [[ra,dec],[ra,dec]], ... ],
#             "boundaries": [ [ [ra,dec], ... ], ... ],
#             "centrum": { "ra": ..., "dec": ... } }, ... ] }
# -----------------------
try:
    with open(CONSTELLATIONS_FILE, 'r') as f:
        const_data = json.load(f)
    constellations = const_data.get("data", [])
    print(f"Constellations chargées : {len(constellations)}")
except FileNotFoundError:
    constellations = []
    print(f"[WARN] Fichier constellations introuvable: {CONSTELLATIONS_FILE} (constellations ignorées)")

# -----------------------
# Visibilité
# -----------------------
def is_visible_star(dec, latitude=LATITUDE):
    return dec > (latitude - 90)

visible_stars = [
    star for star in stars
    if is_visible_star(star['dec']) and star['V'] <= MAX_MAGNITUDE
]

# -----------------------
# Extraction nom
# -----------------------
def extract_star_name(ids_string):
    if not ids_string:
        return None

    parts = ids_string.split('|')
    for p in parts:
        if p.startswith("NAME "):
            return p.replace("NAME ", "").strip()
    return None

# -----------------------
# Projection (RA/Dec -> x/y)
# -----------------------
def project_star(ra_deg, dec_deg):
    # repère actuel: H = -RA
    H = np.radians(-ra_deg)
    r = 90 - dec_deg
    x = r * np.sin(H)
    y = -r * np.cos(H)
    return x, y

def unwrap_ra_pair(ra1, ra2):
    """Ajuste ra2 (±360) pour minimiser |ra2-ra1|."""
    d = ra2 - ra1
    if d > 180:
        ra2 -= 360
    elif d < -180:
        ra2 += 360
    return ra1, ra2

# -----------------------
# Tracé constellations (aster + boundaries + labels)
# -----------------------
def draw_asterisms(ax, aster_segments, linewidth=0.55, alpha=0.60, color='black'):
    """
    aster_segments: [ [[ra,dec],[ra,dec]], ... ]
    """
    for seg in aster_segments:
        if not seg or len(seg) < 2:
            continue
        (ra1, dec1), (ra2, dec2) = seg[0], seg[1]

        ra1, ra2 = unwrap_ra_pair(ra1, ra2)

        x1, y1 = project_star(ra1, dec1)
        x2, y2 = project_star(ra2, dec2)

        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha, color=color)

def draw_boundaries(ax, boundary_loops, linewidth=0.45, alpha=0.35, color='black', linestyle='--'):
    """
    boundary_loops: [ [ [ra,dec], [ra,dec], ... ], [ ... ], ... ]
    """
    for loop in boundary_loops:
        if not loop or len(loop) < 2:
            continue

        xs, ys = [], []
        prev_ra = None

        for ra, dec in loop:
            if prev_ra is not None:
                prev_ra, ra = unwrap_ra_pair(prev_ra, ra)

            x, y = project_star(ra, dec)
            xs.append(x)
            ys.append(y)
            prev_ra = ra

        ax.plot(xs, ys, linewidth=linewidth, alpha=alpha, color=color, linestyle=linestyle)

def draw_constellation_label(ax, centrum, text, fontsize=8, alpha=0.55):
    if not centrum or "ra" not in centrum or "dec" not in centrum:
        return
    x, y = project_star(centrum["ra"], centrum["dec"])
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', alpha=alpha)

# -----------------------
# Horizon
# -----------------------
H_vals = np.linspace(0, 2*np.pi, 1000)
delta_hor = np.degrees(
    np.arctan(-np.cos(H_vals) / np.tan(phi))
)

r_hor = 90 - delta_hor
x_hor = r_hor * np.sin(H_vals)
y_hor = -r_hor * np.cos(H_vals)

max_radius = np.max(r_hor)

# -----------------------
# Création dossier
# -----------------------
os.makedirs("output", exist_ok=True)

# -----------------------
# Rotation étiquettes
# -----------------------
def get_tangent_rotation(x, y):
    angle = np.degrees(np.arctan2(y, x))
    # direction tangentielle
    rotation = angle + 90
    return rotation

# -----------------------
# PDF A4 optimisé
# -----------------------
with PdfPages("output/starmap.pdf") as pdf:

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.85])

    # Cercle global (bordure carte)
    circle = plt.Circle((0, 0), max_radius, fill=False, linewidth=1, color='black')
    ax.add_patch(circle)

    # Horizon
    ax.plot(x_hor, y_hor, linewidth=1)

    # -----------------------
    # Constellations (dessinées sous les étoiles)
    # -----------------------
    if constellations:
        for c in constellations:
            aster = c.get("aster", [])
            boundaries = c.get("boundaries", c.get("boundary", []))  # tolérance si "boundary"
            centrum = c.get("centrum", None)

            # Traits
            if aster:
                draw_asterisms(ax, aster, linewidth=0.55, alpha=0.60, color='black')

            # Limites
            if DRAW_CONSTELLATION_BOUNDARIES and boundaries:
                draw_boundaries(ax, boundaries, linewidth=0.45, alpha=0.30, color='black', linestyle='--')

            # Nom au centre (full name)
            if DRAW_CONSTELLATION_LABELS and c.get("name"):
                draw_constellation_label(ax, centrum, c["name"], fontsize=8, alpha=0.50)

    # -----------------------
    # Étoiles
    # -----------------------
    for star in visible_stars:

        x, y = project_star(star['ra'], star['dec'])

        size = max(1, 4 - star['V'])
        if (extract_star_name(star.get('ids')) == "Lodestar"):
            ax.scatter(x, y, s=size**2, color='red')
            ax.text(
                x + 3,
                y + 3,
                "Polaris",
                fontsize=8,
                rotation=get_tangent_rotation(x, y),
                rotation_mode='anchor',
                ha='center',
                va='center'
            )
        else:
            ax.scatter(x, y, s=size**2, color='black')

        # Affichage nom étoiles principales
        if star['V'] <= LABEL_MAG_LIMIT:
            name = extract_star_name(star.get('ids'))
            if name:
                rotation = get_tangent_rotation(x, y)

                ax.text(
                    x * 1.07,
                    y * 1.07,
                    name,
                    fontsize=8,
                    rotation=rotation,
                    rotation_mode='anchor',
                    ha='center',
                    va='center'
                )

    # ==========================
    # Dates (mois + jours) façon "CLEA"
    # ==========================
    months = [
        "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
        "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
    ]

    # Année non bissextile (suffisant pour un planisphère)
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def day_of_year(month_index_0, day):
        """month_index_0: 0=janvier, day: 1..31"""
        return sum(MONTH_DAYS[:month_index_0]) + day

    # Référence CLEA classique : équinoxe de printemps ~ 21 mars => RA☉ ≈ 0h
    REF_DOY = day_of_year(2, 21)  # 21 mars

    SIDEREAL_YEAR = 365.2422  # jours

    def sun_ra_hours_from_doy(doy):
        """
        Approximation : RA du Soleil progresse ~24h par année sidérale.
        Suffisant pour positionnement planisphère "commercial/éducatif".
        """
        delta = (doy - REF_DOY) / SIDEREAL_YEAR
        ra = (24.0 * delta) % 24.0
        return ra

    def place_on_ring_from_ra(R, ra_hours):
        """
        Conversion RA -> position sur anneau en utilisant
        exactement le même repère que tes étoiles:
        H = -RA ; x = R*sin(H) ; y = -R*cos(H)
        """
        H = -np.radians(ra_hours * 15.0)
        x = R * np.sin(H)
        y = -R * np.cos(H)
        return x, y

    # Rayons d’affichage
    R_months = max_radius * 1.10
    R_days   = max_radius * 1.06

    # Jours à afficher
    days_to_show = [5, 10, 15, 20, 25]

    for m_idx, m_name in enumerate(months):
        # --- Mois placé au 15 ---
        doy_m = day_of_year(m_idx, 15)
        ra_m = sun_ra_hours_from_doy(doy_m)
        x_m, y_m = place_on_ring_from_ra(R_months, ra_m)

        rot_m = get_tangent_rotation(x_m, y_m) + 180

        ax.text(
            x_m, y_m,
            m_name,
            fontsize=10,
            rotation=rot_m,
            rotation_mode='anchor',
            ha='center',
            va='center'
        )

        # --- Jours 5,10,15,20,25 ---
        for d in days_to_show:
            if d > MONTH_DAYS[m_idx]:
                continue

            doy_d = day_of_year(m_idx, d)
            ra_d = sun_ra_hours_from_doy(doy_d)
            x_d, y_d = place_on_ring_from_ra(R_days, ra_d)

            rot_d = get_tangent_rotation(x_d, y_d) + 180

            ax.text(
                x_d, y_d,
                str(d),
                fontsize=7,
                rotation=rot_d,
                rotation_mode='anchor',
                ha='center',
                va='center'
            )

    # Mise à l’échelle
    ax.set_xlim(-max_radius * 1.15, max_radius * 1.15)
    ax.set_ylim(-max_radius * 1.15, max_radius * 1.15)

    ax.set_aspect('equal')
    ax.axis('off')

    # Titre
    fig.text(
        0.5, 0.85,
        f"Carte du ciel - Latitude {LATITUDE}°",
        ha='center',
        fontsize=16
    )

    pdf.savefig(fig)
    plt.close()