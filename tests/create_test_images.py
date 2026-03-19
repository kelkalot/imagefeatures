"""Create test images with known, distinct visual properties for verification."""
import numpy as np
from PIL import Image
import os

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)


def save(name, arr):
    Image.fromarray(arr.astype(np.uint8)).save(os.path.join(OUT, name))
    print(f"  Created {name} ({arr.shape[1]}x{arr.shape[0]})")


# 1. Solid red image — should have trivial color histogram, zero texture
red = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)
save("solid_red.png", red)

# 2. Solid blue — same structure as red, useful for color distance test
blue = np.full((200, 200, 3), [0, 0, 255], dtype=np.uint8)
save("solid_blue.png", blue)

# 3. Red-blue split (left/right) — distinct layout, bimodal color
split = np.zeros((200, 200, 3), dtype=np.uint8)
split[:, :100] = [255, 0, 0]
split[:, 100:] = [0, 0, 255]
save("red_blue_split.png", split)

# 4. Horizontal black-white stripes — strong horizontal edge, high contrast texture
stripes_h = np.zeros((200, 200, 3), dtype=np.uint8)
for i in range(0, 200, 20):
    stripes_h[i:i+10, :] = 255
save("stripes_horizontal.png", stripes_h)

# 5. Vertical black-white stripes — strong vertical edge (compare with horizontal)
stripes_v = np.zeros((200, 200, 3), dtype=np.uint8)
for i in range(0, 200, 20):
    stripes_v[:, i:i+10] = 255
save("stripes_vertical.png", stripes_v)

# 6. Diagonal stripes — tests diagonal edge detection
diag = np.zeros((200, 200, 3), dtype=np.uint8)
for y in range(200):
    for x in range(200):
        if (x + y) % 40 < 20:
            diag[y, x] = 255
save("stripes_diagonal.png", diag)

# 7. Smooth gradient (left=black, right=white) — low texture, smooth layout
gradient = np.zeros((200, 200, 3), dtype=np.uint8)
for x in range(200):
    gradient[:, x] = int(255 * x / 199)
save("gradient_bw.png", gradient)

# 8. Color wheel / rainbow — tests hue distribution
rainbow = np.zeros((200, 200, 3), dtype=np.uint8)
for x in range(200):
    hue = x / 200.0  # 0 to 1
    # HSV to RGB (S=1, V=1)
    h6 = hue * 6
    i = int(h6) % 6
    f = h6 - int(h6)
    if i == 0:
        r, g, b = 255, int(255*f), 0
    elif i == 1:
        r, g, b = int(255*(1-f)), 255, 0
    elif i == 2:
        r, g, b = 0, 255, int(255*f)
    elif i == 3:
        r, g, b = 0, int(255*(1-f)), 255
    elif i == 4:
        r, g, b = int(255*f), 0, 255
    else:
        r, g, b = 255, 0, int(255*(1-f))
    rainbow[:, x] = [r, g, b]
save("rainbow.png", rainbow)

# 9. Checkerboard — high frequency texture, symmetric
checker = np.zeros((200, 200, 3), dtype=np.uint8)
for y in range(200):
    for x in range(200):
        if (x // 25 + y // 25) % 2 == 0:
            checker[y, x] = 255
save("checkerboard.png", checker)

# 10. Natural-like: concentric circles — tests shape features
circles = np.zeros((200, 200, 3), dtype=np.uint8)
cy, cx = 100, 100
for y in range(200):
    for x in range(200):
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if int(dist / 15) % 2 == 0:
            circles[y, x] = 255
save("concentric_circles.png", circles)

print(f"\nCreated 10 test images in {OUT}/")
