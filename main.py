import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LATITUDE = np.radians(45)
MAX_MAGNITUDE = 4
OUTPUT_FILE = 'output/starmap.pdf'
RAYON = 1

with open('stars.json', 'r') as f:
    data = json.load(f)
stars = data['data'] # Lite des étoiles

print(f"Nombre total d'étoiles : {len(stars)}")

def is_visible_star(dec, latitude=LATITUDE):
    # Détermine si une étoile fait partie de la carte du ciel en fonction de sa DEC
    return dec > (latitude - 90)

visible_stars = [star for star in stars if is_visible_star(star['dec'])]
print(f"Nombre d'étoiles visibles : {len(visible_stars)}")


t_values = np.linspace(0, 2 * np.pi, 360)
x_equatorial = RAYON * np.cos(t_values)
y_equatorial = RAYON * np.sin(t_values)

# Générer les points de l'horizon
alpha = np.arctan2(np.sin(t_values) * np.sin(LATITUDE), np.cos(t_values))
delta_proj = np.arcsin(np.sin(t_values) * np.cos(LATITUDE))
# Calculer r' pour l'horizon
r_prime = RAYON * np.cos(delta_proj)
# Convertir en coordonnées cartésiennes pour l'horizon
x_horizon = r_prime * np.cos(alpha)
y_horizon = r_prime * np.sin(alpha)
# Trouver le point le plus bas de l'horizon
y_horizon_min = np.min(y_horizon)
# Le point le plus bas du cercle équatorial
y_equatorial_min = -RAYON
# Calculer le décalage vertical nécessaire
vertical_shift = y_equatorial_min - y_horizon_min
# Appliquer le décalage vertical à l'horizon
y_horizon_shifted = y_horizon + vertical_shift


# Créer un fichier PDF
with PdfPages(OUTPUT_FILE) as pdf:
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=100)  # Format A4 en pouces

    # Tracer le cercle équatorial
    ax.plot(x_equatorial, y_equatorial, label='Équateur Céleste', color='blue')

    # Tracer l'horizon décalé
    ax.plot(x_horizon, y_horizon_shifted, label='Horizon Décalé', color='orange')

    # Placer les étoiles visibles
    for star in visible_stars:
        ra = star['ra']
        dec = star['dec']
        r_star = RAYON * (1 - dec / 90)
        theta_star = np.radians(ra)
        x_star = r_star * np.sin(theta_star)
        y_star = -r_star * np.cos(theta_star)  # Moins pour respecter l'orientation
        ax.scatter(x_star, y_star, color='red', s=2)  # Taille des points pour les étoiles

    ax.set_aspect('equal')
    ax.set_title('Carte du Ciel Visible depuis 45°N')
    ax.legend()

    pdf.savefig(fig)
    plt.close()