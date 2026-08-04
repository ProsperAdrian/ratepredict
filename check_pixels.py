from PIL import Image
import numpy as np

img = Image.open('screenshot.png').convert('RGB')
arr = np.array(img)
# The tabs are probably near the top. Let's look for black pixels (0,0,0) or dark grey
black_pixels = np.where((arr[:,:,0] < 30) & (arr[:,:,1] < 30) & (arr[:,:,2] < 30))
print(f"Found {len(black_pixels[0])} dark pixels")
