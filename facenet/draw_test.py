from PIL import Image, ImageDraw
import face_recognition as fr
import pickle

pkl_path = '/home/enningxie/Documents/DataSets/face_rec/pkl_images/10600.pkl'
image_path = '/home/enningxie/Documents/DataSets/face_images/10600.jpg'


image_data = fr.load_image_file(image_path)
pil_image = Image.fromarray(image_data)
draw = ImageDraw.Draw(pil_image)
with open(pkl_path, 'rb') as f:
    locations = pickle.load(f)

print(len(locations))

print(locations[:3])

for num, location in enumerate(locations):
    top, right, bottom, left = location
    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(str(num))
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), str(num), fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

pil_image.save('./test.jpg')
