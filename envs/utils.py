from envs import settings
from envs.Entity import Entity



def distance(a: Entity, b: Entity):
    return max(abs(a.x - b.x) + abs(a.y - b.y), 1)

def distance_square(a: Entity, b: Entity):
    dist = distance(a, b)
    return dist * dist

def distance_real(a: Entity, b: Entity):
    return distance(a, b) * settings.size_length

def distance_square_real(a: Entity, b: Entity):
    dist = distance(a, b) * settings.size_length
    return dist * dist
