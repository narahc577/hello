import cv2
import numpy as np

# Load the image
img = cv2.imread('image1.png', 0)

# Set the probabilities for salt and pepper noise
salt_prob = 0.1 # Adjust this to change the amount of salt noise
pepper_prob = 0.1  # Adjust this to change the amount of pepper noise

# Create a copy of the image to add noise to
noisy_img = img.copy()

# Get the number of rows and columns of the image
rows, cols = img.shape

# Add salt noise (white pixels)
num_salt = np.ceil(salt_prob * img.size)
x_coords = np.random.randint(0, rows, int(num_salt))
y_coords = np.random.randint(0, cols, int(num_salt))
noisy_img[x_coords, y_coords] = 255

# Add pepper noise (black pixels)
num_pepper = np.ceil(pepper_prob * img.size)
x_coords = np.random.randint(0, rows, int(num_pepper))
y_coords = np.random.randint(0, cols, int(num_pepper))
noisy_img[x_coords, y_coords] = 0

# Save the noisy image
cv2.imwrite('noisy_image.png', noisy_img)

# Print the total number of noisy pixels added
print(f"Total number of noisy pixels added: {int(num_salt + num_pepper)}")
