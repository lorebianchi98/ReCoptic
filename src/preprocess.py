def crop_image(img, crop_percentage=0.1):
    # Get the width and height of the image
    width, height = img.size

    # Calculate the 20% to crop from each side
    left = width * crop_percentage
    top = height * crop_percentage
    right = width * (1- crop_percentage)
    bottom = height * (1- crop_percentage)

    return img.crop((left, top, right, bottom))