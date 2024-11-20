'''
Library for image processing
Date created: 11/15/2024
@valinteshleypierre
'''

# Import libraries
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

model_path = 'face_landmarker_v2_with_blendshapes.task'
# STEP 1: Create an FaceLandmarker object.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)
detector = FaceLandmarker.create_from_options(options)

# def detect_face_and_eyes(image):
#     '''
#     Detects face and eye landmarks using mediapipe and returns landmarks.

#     Parameters:
#         image(numpy.ndarray): Input image

#     Returns:
#         annotated_image (numpy.ndarray): Image with landmarks drawn
#         face_landmarks (list): Detected face landmarks
#     '''

#     logging.info("Initializing Mediapipe Face Mesh.")
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

#     logging.info("Processing image for facial landmarks.")
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if results.multi_face_landmarks:
#         logging.info("Face landmarks detected.")
#         for face_landmarks in results.multi_face_landmarks:
#             for idx, lm in enumerate(face_landmarks.landmark):
#                 h, w, _ = image.shape
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
#             else:
#                 logging.warning("No face landmarks detected.")

#     cv2.imshow("original", image)
#     cv2.imshow("marked", results.multi_face_landmarks)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return image, results.multi_face_landmarks

def draw_landmarks_on_image(image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
    return annotated_image

# def enhance_iamge(image):
#     '''
#     Adjust the input image brightness, contrast, and reducing noise.
#     '''
#     logging.info("Enhancing image: adjusting brightness, contrast, and reducing noise.")

#     # Brightness and contrast adjustment
#     enhanced_image = cv2.convertScaleAbs(image, alpha=1.3, beta=20)

#     # Noise reduction
#     denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
#     logging.info("Image enhancement complete.")
#     return denoised_image

def convert_to_bgr(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    logging.info(f"Original image shape: {image.shape}")

    # Check and convert to BGR if necessary
    if len(image.shape) == 2:  # Grayscale
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        logging.info("Converted grayscale to BGR.")
    elif image.shape[2] == 4:  # RGBA
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        logging.info("Converted RGBA to BGR.")
    else:
        image_bgr = image  # Already BGR
        logging.info("Image is already BGR.")

    # Verify it's a NumPy array
    if not isinstance(image_bgr, np.ndarray):
        image_bgr = np.array(image_bgr)
        logging.info("Converted to NumPy ndarray.")

    logging.info(f"Final NumPy array shape: {image_bgr.shape}, dtype: {image_bgr.dtype}")
    return image_bgr

# Example usage
image_path = "sample_images/tesh_profile_crop.jpg"
bgr_image = convert_to_bgr(image_path)

# Main function
def main(detector):
    '''
    Main function to run the face and eye detection pipeline
    '''

    # Test messages

    # print(f"Current working directory: {os.getcwd()}") # check that you're in the right working directory
    logging.info("Starting program.")
    # # Parse command-line areguments

    # load the image
    # Check to see if the program is reading the right image
    image_path = "/Users/teshpierre/Documents/Programming/aEYE_health/sample_images/tesh_profile_crop.jpg" # "image-2.png"
    mp_image = mp.Image.create_from_file(image_path)

    # Detect face landmarks from the input image.
    detection_result = detector.detect(mp_image)

    mp_image = convert_to_bgr(image_path) # convert the image sto BGR (mp landmark is expecting np.array from bgr files)
    print("--------------------------------------------------", mp_image)

    # Visulaize detection
    annotated_image = draw_landmarks_on_image(mp_image, detection_result)
    cv2.imshow("Image", annotated_image) # cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # if not os.path.exists(image_path):
    #     logging.error(f"File not found: {image_path}")
    # else:
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         logging.warning(f"Unable to load image. Check the file format or OpenCV installation.")
    #     else:
    #         logging.info("Image loaded successfully.")
    #         cv2.imshow("Image", image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # # logging.info(f"Loading image from path: {image_path}")
    # image = mp.Image.create_from_file("image-2.png")
    # # image = cv2.imread(image_path)
    # # cv2.imshow("Image", image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # if image is None:
    #     logging.error(f"Cannot load image from path: {image_path}")
    #     return
    
    # # # Detect face and eyes
    # logging.info("Detecting face and eyes in the image.")
    # # annotated_image, face_landmarks = detect_face_and_eyes(image)
    # annotated_image = draw_landmarks_on_image(image)
    # cv2.imshow("marked", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(detector)
    print('Done')