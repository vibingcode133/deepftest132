# -*- coding: utf-8 -*-
import os
import glob
import pickle
import numpy as np
import torch
from PIL import Image, ImageDraw
import pandas as pd

# Global Imports
from deepface import DeepFace
from deepface.modules import verification as dst

# --- Helper Functions for Image Conversion ---
def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Converts a torch tensor to a PIL Image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- Common Function for Building the Database ---
def _build_db_logic(directory_path, db_save_path, model_name, detector_backend, force_rebuild, force_cpu=False):
    if force_cpu:
        print("Forcing TensorFlow to use CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    
    if force_cpu:
        tf.config.set_visible_devices([], 'GPU')

    if not os.path.isdir(directory_path):
        return (None, f"Error: Directory not found at {directory_path}")
    
    output_dir = os.path.dirname(db_save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not force_rebuild and os.path.exists(db_save_path):
        try:
            with open(db_save_path, 'rb') as f:
                all_embeddings_data = pickle.load(f)
            status = f"Loaded {len(all_embeddings_data)} faces from DB."
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
                    'target_image': os.path.basename(img_path),
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
    else:
        status = "No faces were detected in any images."

    return (all_embeddings_data, status)

# --- Node 1a: Build Database on CPU ---
class FaceDB_BuildEmbeddings_CPU:
    MODEL_OPTIONS = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace"]
    DETECTOR_OPTIONS = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "directory_path": ("STRING", {"default": "/data/app/input/target_faces"}),
                "db_save_path": ("STRING", {"default": "/data/app/output/face_embeddings_db_cpu.pkl"}),
                "model_name": (cls.MODEL_OPTIONS, {"default": "Facenet512"}),
                "detector_backend": (cls.DETECTOR_OPTIONS, {"default": "mtcnn"}),
                "force_rebuild": ("BOOLEAN", {"default": False}),
            } }
    RETURN_TYPES = ("FACE_DB", "STRING",)
    RETURN_NAMES = ("face_database", "status_text",)
    FUNCTION = "build_db"
    CATEGORY = "FaceID"
    def build_db(self, **kwargs):
        return _build_db_logic(**kwargs, force_cpu=True)

# --- Node 1b: Build Database on GPU ---
class FaceDB_BuildEmbeddings_GPU:
    MODEL_OPTIONS = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace"]
    DETECTOR_OPTIONS = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                "directory_path": ("STRING", {"default": "/data/app/input/target_faces"}),
                "db_save_path": ("STRING", {"default": "/data/app/output/face_embeddings_db_gpu.pkl"}),
                "model_name": (cls.MODEL_OPTIONS, {"default": "Facenet512"}),
                "detector_backend": (cls.DETECTOR_OPTIONS, {"default": "mtcnn"}),
                "force_rebuild": ("BOOLEAN", {"default": False}),
            } }
    RETURN_TYPES = ("FACE_DB", "STRING",)
    RETURN_NAMES = ("face_database", "status_text",)
    FUNCTION = "build_db"
    CATEGORY = "FaceID"
    def build_db(self, **kwargs):
        return _build_db_logic(**kwargs, force_cpu=False)


# --- Node 2: Find Matching Faces ---
class FaceDB_FindMatches:
    @classmethod
    def INPUT_TYPES(cls):
        detector_options = FaceDB_BuildEmbeddings_CPU.DETECTOR_OPTIONS
        return {
            "required": {
                "face_database": ("FACE_DB",),
                "source_image": ("IMAGE",),
                "detector_backend": (detector_options, {"default": "mtcnn"}),
                "similarity_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "top_n": ("INT", {"default": 10, "min": 1, "max": 1000}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("best_match_image", "match_image_with_box", "top_n_results_text", "all_filtered_results_text",)
    FUNCTION = "find_matches"
    CATEGORY = "FaceID"

    def find_matches(self, face_database, source_image, detector_backend, similarity_threshold, top_n):
        if not face_database:
            no_match_text = "Error: Face database is empty or invalid."
            return (None, None, no_match_text, no_match_text)

        model_name = face_database[0].get('model_name', 'Facenet512')
        source_pil = tensor2pil(source_image)
        source_np = np.array(source_pil)

        try:
            source_representations = DeepFace.represent(
                img_path=source_np, model_name=model_name,
                detector_backend=detector_backend, enforce_detection=True
            )
            if not source_representations:
                 no_match_text = "Error: No face detected in source image."
                 return (None, None, no_match_text, no_match_text)
            source_embedding = source_representations[0]['embedding']
        except Exception as e:
            error_text = f"Error processing source image: {e}"
            return (None, None, error_text, error_text)

        matches = []
        for target_data in face_database:
            distance = dst.find_cosine_distance(np.array(source_embedding), np.array(target_data['embedding']))
            similarity = (1 - distance) * 100
            if similarity >= similarity_threshold:
                match_info = target_data.copy()
                match_info['similarity_percentage'] = similarity
                match_info['distance'] = distance
                matches.append(match_info)

        if not matches:
            no_match_text = f"No matches found above {similarity_threshold:.1f}% threshold."
            return (None, None, no_match_text, no_match_text)

        sorted_matches_df = pd.DataFrame(matches).sort_values(by='similarity_percentage', ascending=False).reset_index(drop=True)
        
        top_n_df = sorted_matches_df.head(top_n)
        top_n_results_text = f"--- Top {len(top_n_df)} Matches ---\n" + top_n_df[['target_image', 'similarity_percentage', 'distance']].to_string()
        
        all_filtered_results_text = f"--- All {len(sorted_matches_df)} Matches > {similarity_threshold:.1f}% ---\n" + sorted_matches_df[['target_image', 'similarity_percentage', 'distance']].to_string()
        
        # --- THIS IS THE NEW PART: PRINT RESULTS TO THE CONSOLE LOG ---
        print("\n\n==============================================")
        print("          FACEID SEARCH RESULTS             ")
        print("==============================================")
        print(all_filtered_results_text)
        print("==============================================\n\n")
        # -------------------------------------------------------------
        
        best_match = sorted_matches_df.iloc[0].to_dict()
        
        img_pil = Image.open(best_match['image_path']).convert("RGB")
        img_with_box = img_pil.copy()
        draw = ImageDraw.Draw(img_with_box)
        
        facial_area = best_match['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        draw.rectangle([x, y, x+w, y+h], outline="lime", width=5)
        
        return (pil2tensor(img_pil), pil2tensor(img_with_box), top_n_results_text, all_filtered_results_text)


# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "FaceDB_BuildEmbeddings_CPU": FaceDB_BuildEmbeddings_CPU,
    "FaceDB_BuildEmbeddings_GPU": FaceDB_BuildEmbeddings_GPU,
    "FaceDB_FindMatches": FaceDB_FindMatches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDB_BuildEmbeddings_CPU": "Build Face DB (CPU)",
    "FaceDB_BuildEmbeddings_GPU": "Build Face DB (GPU)",
    "FaceDB_FindMatches": "Find Face in Database",
}