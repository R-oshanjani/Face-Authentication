from fastapi import FastAPI, UploadFile, File, HTTPException
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import io

app = FastAPI(title="Face Authentication")

device = torch.device("cpu")

mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def load_image(file: UploadFile):
    try:
        data = file.file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")


def get_face_embedding(img: Image.Image):
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return None, None

    faces = mtcnn(img)
    if faces is None:
        return None, None

    faces = faces.to(device)

    with torch.no_grad():
        emb = facenet(faces)[0]

    emb = emb / emb.norm()

    x1, y1, x2, y2 = boxes[0]
    box = {
        "top": int(y1),
        "right": int(x2),
        "bottom": int(y2),
        "left": int(x1),
    }

    return emb, box


def cosine_similarity(v1, v2):
    return float(torch.dot(v1, v2).item())


@app.get("/")
def root():
    return {"message": "Face Verification API is running"}


@app.post("/verify-faces")
async def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):

    valid = {"image/jpeg", "image/png", "image/jpg"}
    if image1.content_type not in valid or image2.content_type not in valid:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images allowed")

    img1 = load_image(image1)
    img2 = load_image(image2)

    emb1, box1 = get_face_embedding(img1)
    emb2, box2 = get_face_embedding(img2)

    if emb1 is None:
        raise HTTPException(status_code=400, detail="No face found in image1")
    if emb2 is None:
        raise HTTPException(status_code=400, detail="No face found in image2")

    score = cosine_similarity(emb1, emb2)
    threshold = 0.7
    result = "same person" if score >= threshold else "different person"

    return {
        "verification_result": result,
        "similarity_score": score,
        "image1_box": box1,
        "image2_box": box2,
    }
