'''
Library for image processing
Date created: 11/15/2024
@valinteshleypierre
'''

# Import libraries
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(messages)s",
    level=logging.INFO
)

def detect_face_and_eyes(image):
    '''
    Detects face and eye landmarks using mediapipe and returns landmarks.

    Parameters:
        image(numpy.ndarray): Input image

    Returns:
        annotated_image (numpy.ndarray): Image with landmarks drawn
        face_landmarks (list): Detected face landmarks
    '''

    logging.info("Initializing Mediapipe Face Mesh.")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    logging.info("Processing image for facial landmarks.")
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        logging.info("Face landmarks detected.")
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                h, w, _ = image.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
            else:
                logging.warning("No face landmarks detected.")
    return image, results.multi_face_landmarks

def enhance_iamge(image):
    '''
    Adjust the input image brightness, contrast, and reducing noise.
    '''
    logging.info("Enhancing image: adjusting brightness, contrast, and reducing noise.")

    # Brightness and contrast adjustment
    enhanced_image = cv2.convertScaleAbs(image, alpha=1.3, beta=20)

    # Noise reduction
    denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    logging.info("Image enhancement complete.")
    return denoised_image


# Main function
def main():
    logging.info("Starting program.")
    # Parse command-line areguments

    # load the image
    image_path = "sample_images/tesh_profile_crop.jpg"
    
    return

if __name__ == "__main__":
    main()
    print('Done')