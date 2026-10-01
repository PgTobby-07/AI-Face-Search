import cv2
import numpy as np
from insightface.app import FaceAnalysis


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

THRESHOLD = 0.50


# --------------------------------------------------
# LOAD FACE RECOGNITION MODEL
# --------------------------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# --------------------------------------------------
# GET FACE EMBEDDINGS
# --------------------------------------------------

def get_face_embeddings(image_path):
    """
    Detect every face in an image and return
    an embedding for each detected face.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return []

    faces = app.get(image)

    embeddings = []

    for face in faces:

        embedding = face.embedding

        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)

    return embeddings


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(embedding1, embedding2):
    """
    Measure how similar two face embeddings are.
    """

    return np.dot(embedding1, embedding2)


# --------------------------------------------------
# IDENTIFY ONE FACE
# --------------------------------------------------

def identify_face(face_embedding, people):
    """
    Compare one detected face against every
    reference embedding.

    Returns:
        person name
        similarity score
    """

    best_person = None
    best_score = -1

    # Go through every person
    for person_name, reference_embeddings in people.items():

        # Go through every reference photo
        # belonging to that person
        for reference_embedding in reference_embeddings:

            similarity = cosine_similarity(
                face_embedding,
                reference_embedding
            )

            if similarity > best_score:

                best_score = similarity
                best_person = person_name

    # Check the threshold
    if best_score >= THRESHOLD:
        return best_person, best_score

    return None, best_score


# --------------------------------------------------
# SEARCH GALLERY
# --------------------------------------------------

def search_photos(reference_people, gallery_images):
    """
    Search uploaded gallery images for the
    people in reference_people.
    """

    results = []

    # Go through every gallery photo
    for image_path in gallery_images:

        # Detect every face in this photo
        faces = get_face_embeddings(image_path)

        photo_matches = []

        # Check every detected face
        for face_number, face_embedding in enumerate(faces):

            person, score = identify_face(
                face_embedding,
                reference_people
            )

            if person is not None:

                photo_matches.append({
                    "person": person,
                    "score": score,
                    "face_number": face_number + 1
                })

        # Only save photos containing
        # at least one recognized person
        if photo_matches:

            results.append({
                "image": image_path,
                "matches": photo_matches
            })

    return results