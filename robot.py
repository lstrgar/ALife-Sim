import numpy as np
from scipy import ndimage

# Size of the binary mask (8x8 grid of voxels)
MASK_DIM = 8 

# Scale of the robot (edge length of a voxel)
# NOTE: this is very important as the simulator physics are configured to use this scale, more or less.
SCALE = 0.1 

def load_robots(num_robots):
    return [sample_robot() for _ in range(num_robots)]

# Randomly sample a binary mask of size MASK_DIM x MASK_DIM
# Convert the binary mask to a mass-spring robot geometry
# The parameter p is by default set to 0.55, which is the probability of a voxel being filled.
# This is a manually tuned value that seems to produce a variety of different robot geometries.
def sample_robot(p=0.55):
    mask = sample_mask(p)
    masses, springs = mask_to_robot(mask)
    masses = masses * SCALE # NOTE: scale of the robot geometry is KEY to stable simulation!
    return {
        "n_masses": masses.shape[0],
        "n_springs": springs.shape[0],
        "masses": masses,
        "springs": springs,
    }

# Convert a voxel position to a list of mass coordinates
# Each voxel has a mass located at each of its four corners
def voxel_to_masses(row, col):
    return [
        [row, col],
        [row, col+1],
        [row+1, col],
        [row+1, col+1],
    ]

# Convert a binary mask to a mass-spring robot geometry
# Each voxel is represented by 4 masses and 6 springs
# Masses are located at the corners of the voxel
# Springs connect adjacent masses along the edges and diagonals of the voxel
def mask_to_robot(mask):
    spring_connections = [
        [0, 1], # bottom left (bl) to bottom right (br)
        [0, 2], # bl to top left (tl)
        [1, 3], # br to top right (tr)
        [2, 3], # tl to tr
        [0, 3], # bl to tr
        [1, 2], # br to tl
    ]
    masses = []
    springs = []
    rows, cols = np.where(mask)
    n_voxels = len(rows)
    for i in range(n_voxels):
        row = rows[i]
        col = cols[i]
        coords = voxel_to_masses(row, col)
        for c in coords:
            if c not in masses: # NOTE: make sure to avoid duplicates!
                masses.append(c)
        for a, b, in spring_connections:
            ca = coords[a]
            cb = coords[b]
            ia = masses.index(ca)
            ib = masses.index(cb)
            s = [min(ia, ib), max(ia, ib)]
            if s not in springs: # NOTE: make sure to avoid duplicates!
                springs.append(s)
    masses = np.array(masses, dtype=np.float32) # Numpy array of shape (n_masses, 2)
    springs = np.array(springs, dtype=np.int32) # Numpy array of shape (n_springs, 2)
    return masses, springs

# Sample a binary mask of size MASK_DIM x MASK_DIM
# Select the largest connected component in the mask
# Zero out the rest of the mask
# Shift the largest component to the bottom left corner of the mask
def sample_mask(p):
    mask = np.random.uniform(0.0, 1.0, size=(MASK_DIM, MASK_DIM))
    mask = mask < p
    labeled, num_features = ndimage.label(mask)
    if num_features == 0: # If the mask is empty, try again
        return sample_mask(p)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = (labeled == largest_component)
    rows, cols = np.where(mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    component = mask[min_row:max_row+1, min_col:max_col+1]
    new_mask = np.zeros((MASK_DIM, MASK_DIM), dtype=int)
    component_height, component_width = component.shape
    new_mask[MASK_DIM - component_height:MASK_DIM, 0:component_width] = component.astype(int)
    return new_mask