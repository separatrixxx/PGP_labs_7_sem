import os
import imageio
from natsort import natsorted

images_dir = "../images"
output_gif = "../animation.gif"

image_files = natsorted(
    [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
)

with imageio.get_writer(output_gif, mode='I', duration=0.5) as writer:
    for img_path in image_files:
        image = imageio.imread(img_path)
        writer.append_data(image)

print(f"GIF создан: {output_gif}")
