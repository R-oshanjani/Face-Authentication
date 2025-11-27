Face Verification API

A simple FastAPI service that checks whether two face images belong to the same person.
The project uses MTCNN for face detection and FaceNet (InceptionResnetV1) for generating face embeddings.

How It Works

Each uploaded image is read and converted to RGB.

MTCNN detects the face and returns the cropped/processed face.

FaceNet generates a 512-dimension embedding for each face.

Cosine similarity is computed between the two embeddings.

If the similarity score is above 0.7, the images are treated as the same person.