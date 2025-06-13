# -*- coding: utf-8 -*-
import os
import glob
import pickle
import numpy as np
import torch
from PIL import Image, ImageDraw

# Suppress TensorFlow logging before importing deepface
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Import deepface directly. If this fails, the whole node will fail, which is clear.
try:
    from deepface import DeepFace
    from deepface.modules import verification as dst
except ImportError:
    raise ImportError("DeepFace or its dependencies could not be imported. Please ensure it is installed correctly via requirements.txt.")

# --- Helper Functions for Image Conversion ---
def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Converts a torch tensor to a PIL Image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# --- Node 1: Build/Load Face Embeddings Database ---
class FaceDB_BuildEmbeddings:
    """
    Scans a directory for images, generates face embeddings for each,
    and saves/loads them from a pickle file. This acts as our face database.
    """
    MODEL_OPTIONS = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace"]
    DETECTOR_OPTIONS = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "/data/app/inputs/target_faces"}), # Adjusted default path
                "db_save_path": ("STRING", {"default": "/data/app/outputs/face_embeddings_db.pkl"}), # Adjusted default path
                "model_name": (cls.MODEL_OPTIONS, {"default": "Facenet512"}),
                "detector_backend": (cls.DETECTOR_OPTIONS, {"default": "mtcnn"}),
                "force_rebuild": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FACE_DB", "STRING",)
    RETURN_NAMES = ("face_database", "status_text",)
    FUNCTION = "build_or_load_db"
    CATEGORY = "FaceID"

    def build_or_load_db(self, directory_path, db_save_path, model_name, detector_backend, force_rebuild):
        if not os.path.isdir(directory_path):
            return (None, f"Error: Directory not found at {directory_path}")
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(db_save_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not force_rebuild and os.path.exists(db_save_path):
            try:
                with open(db_save_path, 'rb') as f:
                    all_embeddings_data = pickle.load(f)
                status = f"Loaded {len(all_embeddings_data)} faces from DB."
                print(status)
                return (all_embeddings_data, status)
            except Exception as e:
                print(f"Could not load DB file, will rebuild. Error: {e}")

        print("Generating new embeddings...")
        all_embeddings_data = []
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
            image_paths.extend(glob.glob(os.path.join(directory_path, ext)))

        if not image_paths:
            return (None, f"No images found in {directory_path}.")

        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            try:
                img_representations = DeepFace.represent(
                    img_path=img_path, model_name=model_name,
                    detector_backend=detector_backend, enforce_detection=True
                )
                for rep_idx, rep_obj in enumerate(img_representations):
                    embedding_data = {
                        'image_path': img_path, 'embedding': rep_obj['embedding'],
                        'facial_area': rep_obj['facial_area'], 'face_index_in_image': rep_idx,
                        'model_name': model_name
                    }
                    all_embeddings_data.append(embedding_data)
            except Exception as e:
                print(f"  Skipping {os.path.basename(img_path)} due to error: {e}")

        if all_embeddings_data:
            with open(db_save_path, 'wb') as f:
                pickle.dump(all_embeddings_data, f)
            status = f"Saved {len(all_embeddings_data)} face embeddings to DB."
            print(status)
        else:
            status = "No faces were detected in any images."

        return (all_embeddings_data, status)

# --- Node 2: Find Matching Faces ---
class FaceDB_FindMatches:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_database": ("FACE_DB",),
                "source_image": ("IMAGE",),
                "detector_backend": (FaceDB_BuildEmbeddings.DETECTOR_OPTIONS, {"default": "mtcnn"}),
                "similarity_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING",)
    RETURN_NAMES = ("best_match_image", "match_image_with_box", "best_match_info",)
    FUNCTION = "find_matches"
    CATEGORY = "FaceID"

    def find_matches(self, face_database, source_image, detector_backend, similarity_threshold):
        if not face_database:
            return (None, None, "Error: Face database is empty.")

        model_name = face_database[0].get('model_name', 'Facenet512')
        source_pil = tensor2pil(source_image)
        source_np = np.array(source_pil)

        try:
            source_representations = DeepFace.represent(
                img_path=source_np, model_name=model_name,
                detector_backend=detector_backend, enforce_detection=True
            )
            if not source_representations:
                 return (None, None, "No face detected in source image.")
            source_embedding = source_representations[0]['embedding']
        except Exception as e:
            return (None, None, f"Error processing source image: {e}")

        matches = []
        for target_data in face_database:
            distance = dst.find_cosine_distance(np.array(source_embedding), np.array(target_data['embedding']))
            similarity = (1 - distance) * 100
            if similarity >= similarity_threshold:
                match_info = target_data.copy()
                match_info['similarity'] = similarity
                matches.append(match_info)

        if not matches:
            return (None, None, f"No matches found above {similarity_threshold}% threshold.")

        best_match = max(matches, key=lambda x: x['similarity'])
        
        img_pil = Image.open(best_match['image_path']).convert("RGB")
        img_with_box = img_pil.copy()
        draw = ImageDraw.Draw(img_with_box)
        x, y, w, h = best_match['facial_area'].values()
        draw.rectangle([x, y, x+w, y+h], outline="lime", width=5)

        info_str = f"Best Match: {os.path.basename(best_match['image_path'])}\nSimilarity: {best_match['similarity']:.2f}%"
        
        return (pil2tensor(img_pil), pil2tensor(img_with_box), info_str)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "FaceDB_BuildEmbeddings": FaceDB_BuildEmbeddings,
    "FaceDB_FindMatches": FaceDB_FindMatches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDB_BuildEmbeddings": "Build/Load Face Database",
    "FaceDB_FindMatches": "Find Face in Database",
}