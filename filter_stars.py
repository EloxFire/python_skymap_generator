import json

LATITUDE = 45
MAX_MAGNITUDE = 6

with open('stars1.json', 'r') as f:
    data = json.load(f)
stars = data['data'] # Lite des étoiles

print(f"Nombre total d'étoiles : {len(stars)}")

def is_visible_star(dec, latitude=LATITUDE):
    # Détermine si une étoile fait partie de la carte du ciel en fonction de sa DEC
    return dec > (latitude - 90)

visible_stars = [star for star in stars if is_visible_star(star['dec'])]
print(f"Nombre d'étoiles visibles : {len(visible_stars)}")

