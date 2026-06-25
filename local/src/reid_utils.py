import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import supervision as sv
import base64
import os
import logging
from confluent_kafka import Consumer, Producer
import torchreid

# torchreid is installed editable from osnet/deep-person-reid (see setup.sh / README),
# so it imports directly with no sys.path hack needed.

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SHARED UTILS ---

def get_kafka_producer(broker):
    return Producer({'bootstrap.servers': broker})

def get_kafka_consumer(broker, group_id, topics):
    c = Consumer({
        'bootstrap.servers': broker,
        'group.id': group_id,
        'auto.offset.reset': 'latest',
        'fetch.min.bytes': 10000 
    })
    c.subscribe(topics)
    return c

def encode_image_base64(img_numpy, quality=60):
    _, buffer = cv2.imencode('.jpg', img_numpy, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

def decode_image_base64(b64_string):
    if not b64_string: return None
    try:
        nparr = np.frombuffer(base64.b64decode(b64_string), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: return None

def merge_truck_boxes(detections, frame_shape):
    """Merge truck/bus boxes that YOLO split in two."""
    if len(detections) < 2: return detections
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
    
    BIG_VEHICLE_CLASSES = [5, 7] # Bus, Truck
    keep_indices = []
    merged_indices = set()
    new_xyxy, new_conf, new_cls, new_ids = [], [], [], []

    sorted_idx = np.argsort(xyxy[:, 0])
    
    for i in range(len(sorted_idx)):
        idx_curr = sorted_idx[i]
        if idx_curr in merged_indices: continue
        
        current_box = xyxy[idx_curr].copy()
        current_cls = class_id[idx_curr]
        
        if current_cls not in BIG_VEHICLE_CLASSES:
            new_xyxy.append(current_box)
            new_conf.append(confidence[idx_curr])
            new_cls.append(current_cls)
            new_ids.append(tracker_id[idx_curr])
            continue

        for j in range(i + 1, len(sorted_idx)):
            idx_next = sorted_idx[j]
            if idx_next in merged_indices: continue
            if class_id[idx_next] not in BIG_VEHICLE_CLASSES: continue
            
            next_box = xyxy[idx_next]
            gap = max(0, next_box[0] - current_box[2])
            y_min = max(current_box[1], next_box[1])
            y_max = min(current_box[3], next_box[3])
            overlap_h = max(0, y_max - y_min)
            min_height = min(current_box[3]-current_box[1], next_box[3]-next_box[1])
            
            if gap < 50 and overlap_h > 0.6 * min_height:
                current_box[0] = min(current_box[0], next_box[0])
                current_box[1] = min(current_box[1], next_box[1])
                current_box[2] = max(current_box[2], next_box[2])
                current_box[3] = max(current_box[3], next_box[3])
                merged_indices.add(idx_next)

        new_xyxy.append(current_box)
        new_conf.append(confidence[idx_curr])
        new_cls.append(current_cls)
        new_ids.append(tracker_id[idx_curr])
        merged_indices.add(idx_curr)

    if not new_xyxy: return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(new_xyxy),
        confidence=np.array(new_conf),
        class_id=np.array(new_cls),
        tracker_id=np.array(new_ids)
    )

# --- FEATURE EXTRACTOR (DYNAMIC PATH) ---

class FeatureExtractor:
    def __init__(self, model_path, model_arch='osnet_x1_0', logger=None):
        self.log = logger or logging.getLogger("reid")
        self.log.info("Init FeatureExtractor",
                      extra={"event": "extractor_init", "arch": model_arch,
                             "device": DEVICE, "weights": model_path})

        self.model = torchreid.models.build_model(
            name=model_arch, num_classes=100, loss='triplet', pretrained=False
        )

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                state_dict = checkpoint.get('state_dict', checkpoint)
                # Drop the classifier head
                filtered_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
                self.model.load_state_dict(filtered_dict, strict=False)
                self.log.info("Loaded OSNet weights",
                              extra={"event": "weights_loaded", "weights": model_path})
            except Exception:
                self.log.error("Failed to load OSNet weights", exc_info=True,
                               extra={"event": "weights_error", "weights": model_path})
        else:
            self.log.warning("OSNet weights not found, using random weights",
                             extra={"event": "weights_missing", "weights": model_path})

        self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_batch(self, crops):
        if not crops: return np.array([])
        batch_tensors = []
        for img in crops:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            batch_tensors.append(self.transform(img_pil))
        
        batch_t = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            features = self.model(batch_t)
            norm = features.norm(p=2, dim=1, keepdim=True)
            features = features.div(norm.expand_as(features))
        return features.cpu().numpy()
    

class MOTResultWriter:
    def __init__(self, output_path, target_width=None, original_width=None,
                 original_height=None, logger=None):
        """
        Write tracking results in MOTChallenge format.
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
        """
        self.log = logger or logging.getLogger("reid")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.output_path = output_path
        self.file = open(output_path, 'w')

        # Compute scale factor when the stream is resized
        self.scale_x = 1.0
        self.scale_y = 1.0

        if target_width and original_width and original_height:
            # Assume aspect ratio is kept on resize
            target_height = int(original_height * (target_width / original_width))
            self.scale_x = original_width / target_width
            self.scale_y = original_height / target_height
            self.log.info("MOTResultWriter coordinate scaling",
                          extra={"event": "mot_scale", "scale_x": round(self.scale_x, 2),
                                 "scale_y": round(self.scale_y, 2), "output": output_path})

    def write(self, frame_idx, track_id, xyxy, conf=1.0):
        """
        frame_idx: int (1-based)
        xyxy: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = xyxy

        # Scale back to the GT's original resolution
        x1 *= self.scale_x
        y1 *= self.scale_y
        x2 *= self.scale_x
        y2 *= self.scale_y

        w = x2 - x1
        h = y2 - y1

        # MOT format: frame, id, left, top, w, h, conf, -1, -1, -1
        line = f"{int(frame_idx)},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
        self.file.write(line)
        self.file.flush()  # write immediately

    def close(self):
        self.file.close()