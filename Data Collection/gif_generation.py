from PIL import Image
import glob
import os

folder = './simulation-tests/realtime-test-5/'  # Change to your folder path
pattern = os.path.join(folder, '*.png')        # Matches all .png files
files = sorted(glob.glob(pattern))
print("Found files:", files)

if files:
    frames = [Image.open(f) for f in files]
    frames[0].save(
        'output2.gif',
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=200,  # milliseconds per frame
        loop=0
    )
    print("GIF saved as gif")
else:
    print("No .png images found in", folder)
