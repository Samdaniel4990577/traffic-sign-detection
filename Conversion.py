import os
from PIL import Image

# Specify the folder containing PNG images
folder_path = 'TrafficSign/images'

# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # Check if the file is a PNG
        # Open the PNG image
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        
        # Convert the image to RGB (necessary for saving as JPEG)
        img = img.convert('RGB')
        
        # Set the output path for the JPEG file
        output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.jpg")
        
        # Save the image as JPEG
        img.save(output_path, "JPEG")
        
        print(f"Converted {filename} to {output_path}")
