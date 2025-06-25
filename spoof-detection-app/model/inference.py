# === COMPLETE MINIMAL INFERENCE SCRIPT ===
# This is everything you need to run inference only

import os
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage, ImageFilter
from transformers import DPTForDepthEstimation
from torchvision import models, transforms
import mediapipe as mp

# Suppress warnings and force load
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


# === MODEL ARCHITECTURE (REQUIRED) ===
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# === INITIALIZATION HELPER ===
def init_models(model_path='./model/improved_deepfake_detector.pth'):
    """
    Load and return:
      - midas depth model
      - two ResNet backbones (texture & patch)
      - trained MLP classifier and its scaler
    """
    # 1) MiDaS
    print("📥 Loading MiDaS depth model...")
    midas = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas"
    ).to(device).eval()

    # 2) ResNet backbones
    def load_resnet_backbone():
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(
            *list(base.children())[:-2],
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        ).eval().to(device)

    print("📥 Loading ResNet backbones...")
    resnet_texture = load_resnet_backbone()
    resnet_patch   = load_resnet_backbone()

    # 3) Your classifier + scaler
    print("📥 Loading trained MLP classifier...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    classifier = ImprovedMLP(checkpoint['feature_dim']).to(device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    scaler = checkpoint['scaler']
    print(f"✅ Loaded classifier expecting {checkpoint['feature_dim']} features")

    return {
        "midas": midas,
        "resnet_texture": resnet_texture,
        "resnet_patch":   resnet_patch,
        "classifier":     classifier,
        "scaler":         scaler,
    }


# === GLOBAL MODELS ===
model_bundle = init_models('model/improved_deepfake_detector.pth')
midas          = model_bundle["midas"]
resnet_tf      = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
depth_tf       = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
resnet_texture = model_bundle["resnet_texture"]
resnet_patch   = model_bundle["resnet_patch"]
detector       = model_bundle["classifier"]
scaler         = model_bundle["scaler"]

# MediaPipe setup
print("📥 Setting up MediaPipe...")
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

FACIAL_PATCHES = {
    "left_eye":   list(range(33, 42)) + list(range(160, 165)),
    "right_eye":  list(range(263, 272)) + list(range(385, 390)),
    "nose":       [1, 2, 5, 6, 19, 20, 98, 99, 327, 328],
    "mouth":      list(range(61, 68)) + list(range(291, 298)) + [13, 14, 17, 18],
    "left_cheek": [116, 117, 118, 119, 120, 121, 126, 142],
    "right_cheek":[345, 346, 347, 348, 349, 350, 355, 371],
    "forehead":   [9, 10, 151, 337, 299, 333, 298, 301]
}


# === FEATURE EXTRACTION FUNCTIONS ===
def extract_texture_feature_cnn(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    sobels = []
    for k in (3, 5):
        sx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=k)
        sobels.append(np.sqrt(sx**2 + sy**2))
    mean_sobel = np.mean(sobels, axis=0).clip(0, 255).astype(np.uint8)
    sobel_rgb  = np.stack([mean_sobel]*3, axis=-1)

    t = resnet_tf(sobel_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        return resnet_texture(t).squeeze()


def crop_patch(image_rgb, landmarks, idxs):
    h, w, _ = image_rgb.shape
    pts = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in idxs
    ])
    if pts.size == 0:
        return None

    x,y,pw,ph = cv2.boundingRect(pts)
    pad = 5
    x, y = max(0, x-pad), max(0, y-pad)
    pw, ph = min(w-x, pw+2*pad), min(h-y, ph+2*pad)
    if pw < 15 or ph < 15:
        return None

    return image_rgb[y:y+ph, x:x+pw]


def extract_patch_feature_aligned(image_rgb):
    res = mp_face.process(image_rgb)
    if not res.multi_face_landmarks:
        return torch.zeros(len(FACIAL_PATCHES)*512).to(device)

    lm = res.multi_face_landmarks[0].landmark
    feats = []
    for _, idxs in FACIAL_PATCHES.items():
        p = crop_patch(image_rgb, lm, idxs)
        if p is None:
            feats.append(torch.zeros(512).to(device))
        else:
            p64 = cv2.resize(p, (64,64))
            t = resnet_tf(p64).unsqueeze(0).to(device)
            with torch.no_grad():
                feats.append(resnet_patch(t).squeeze())
    return torch.cat(feats, dim=0)


def extract_depth_feature(image_rgb, midas_model):
    pil = PILImage.fromarray(image_rgb).filter(ImageFilter.BLUR)
    inp = depth_tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        dm = midas_model(pixel_values=inp).predicted_depth.squeeze()
    mn, mx = dm.min(), dm.max()
    norm = (dm - mn)/(mx - mn) if mx - mn > 1e-8 else torch.zeros_like(dm)

    pools = []
    for s in ((8,8),(16,16)):
        p = F.adaptive_avg_pool2d(norm.unsqueeze(0).unsqueeze(0), s)
        pools.append(p.flatten())
    return torch.cat(pools, dim=0)


def extract_boundary_features(image_rgb):
    res = mp_face.process(image_rgb)
    if not res.multi_face_landmarks:
        return np.zeros(10)

    lm = res.multi_face_landmarks[0].landmark
    h, w, _ = image_rgb.shape
    contours = [10,338,297,332,284,251,389,356,454,323,361,288,
                397,365,379,378,400,377,152,148,176,149,150,136,
                172,58,132,93,234,127,162,21,54,103,67,109]

    pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in contours])
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    kern = np.ones((5,5),np.uint8)
    region = cv2.dilate(mask,kern,3) - cv2.erode(mask,kern,3)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)*(region>0)
    edge_density = edges.sum()/region.sum() if region.sum()>0 else 0

    grads = []
    for c in range(3):
        g = cv2.Sobel(image_rgb[:,:,c],cv2.CV_64F,1,0,3)**2 + \
            cv2.Sobel(image_rgb[:,:,c],cv2.CV_64F,0,1,3)**2
        grads.append(np.sqrt(g)[region>0].mean() if (region>0).any() else 0)
    return np.array([edge_density]+grads+[0]*6)


def extract_combined_features(image_rgb, midas_model):
    tex    = extract_texture_feature_cnn(image_rgb)
    patch  = extract_patch_feature_aligned(image_rgb)
    depth  = extract_depth_feature(image_rgb, midas_model)
    bound  = extract_boundary_features(image_rgb)

    tn = F.normalize(tex.flatten(), dim=0)
    pn = F.normalize(patch.flatten(), dim=0)
    dn = F.normalize(torch.tensor(bound, dtype=torch.float32).to(device), dim=0)
    z  = F.normalize(torch.tensor(depth.flatten(), dtype=torch.float32).to(device), dim=0)

    combined = torch.cat([0.35*tn, 0.40*pn, 0.15*dn, 0.10*z], dim=0)
    return combined.detach().cpu().numpy()


# === SINGLE-IMAGE TEST ===
def test_image(image_path, model, scaler, midas_model, device=device):
    """Test a single image and show summary."""
    if not isinstance(image_path, str):
        raise TypeError(f"image_path must be a string, got {type(image_path)}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_combined_features(rgb, midas_model).reshape(1, -1)
    scaled = scaler.transform(feats)
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze().cpu().item()
    pred = "FAKE" if logit > 0.5 else "REAL"
    rP, fP = (1-logit)*100, logit*100

    print(f"✅ {image_path}: {pred} (Real: {rP:.1f}% | Fake: {fP:.1f}%)")
    return pred, rP, fP


# === MAIN EXECUTION ===
if __name__ == "__main__":
    uploads_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "uploads/train/real")
    )
    if not os.path.isdir(uploads_dir):
        print(f"Uploads directory not found: {uploads_dir}")
    else:
        # Example single-file run
        sample = "../uploads/train/fake/01_02__exit_phone_room__YVGY8LOK_frame84.jpg"
        print("🧪 Running inference on one image...")
        test_image(sample, detector, scaler, midas)
        print("🎉 Inference complete!")
