import cv2
import torch
import numpy as np
import os
import time
from models import MultiModalTokenizer, HierarchicalEncoder, MultiLevelFusionMLP
from downstream import AttentiveSegmentationProbe

class LiveVJEPATracker:
    def __init__(self, checkpoint_path=None, patch_size=16, embed_dim=192, input_res=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_res = input_res
        self.patch_size = patch_size
        self.num_patches_per_side = input_res // patch_size
        
        print(f"⚙️ Launching Multi-Object V-JEPA 2.1 Production Core on: {self.device}")
        
        # Initialize Core Structures
        self.tokenizer = MultiModalTokenizer(patch_size=patch_size, embed_dim=embed_dim).to(self.device)
        self.context_encoder = HierarchicalEncoder(embed_dim=embed_dim, depth=6).to(self.device)
        self.fusion_mlp = MultiLevelFusionMLP(embed_dim=embed_dim).to(self.device)
        
        # Load custom Trained Neural Probe Head
        self.probe = AttentiveSegmentationProbe(embed_dim=embed_dim, num_classes=2).to(self.device)
        if os.path.exists("probe_trained.pth"):
            self.probe.load_state_dict(torch.load("probe_trained.pth", map_location=self.device))
            print("🎯 Successfully loaded optimized weights: probe_trained.pth")
        else:
            raise FileNotFoundError("Could not locate trained probe weights file.")
        
        self.tokenizer.eval()
        self.context_encoder.eval()
        self.fusion_mlp.eval()
        self.probe.eval()

        # --- PATH 2: MULTI-OBJECT DICTIONARY MEMORY ---
        self.registry = {}  # Format: {class_id: {"vector": tensor, "label": str, "color": tuple, "kalman": KF, "initialized": bool}}
        
        # UI Setup for slots
        self.slot_configs = {
            ord('1'): {"id": 1, "label": "USER", "color": (0, 255, 0)},       # Green
            ord('2'): {"id": 2, "label": "OBJECT_A", "color": (255, 255, 0)}, # Cyan
            ord('3'): {"id": 3, "label": "OBJECT_B", "color": (255, 0, 255)}  # Purple
        }

        # Action Trigger Baseline Registry (Specifically bound to User Class ID 1)
        self.calibrated_cx = None
        self.calibrated_area = None
        self.last_action_time = 0
        self.cooldown_period = 1.5
        self.calibration_frames_gathered = 0
        self.calibration_cx_accumulator = []
        self.calibration_area_accumulator = []
        self.is_calibrating = False

    def build_new_kalman_filter(self):
        """Generates an independent tracker filter state per registered object slot."""
        kf = cv2.KalmanFilter(4, 4)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.5
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def process_and_track(self, frame):
        orig_h, orig_w, _ = frame.shape
        current_time = time.time()
        
        # 1. Forward Pass Inference
        resized = cv2.resize(frame, (self.input_res, self.input_res))
        rgb_canvas = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_canvas).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            tokens = self.tokenizer(tensor, modality="image")
            dense_features = self.context_encoder(tokens)
            if isinstance(dense_features, list): dense_features = dense_features[-1]
            log_probs = torch.log_softmax(self.probe(dense_features), dim=-1).squeeze(0)
            target_log_probs = log_probs[:, 1].cpu().numpy()
            flat_features = dense_features.squeeze(0)

        # 2. Key Registry Listener (Slots 1, 2, or 3)
        key = cv2.waitKey(1) & 0xFF
        if key in self.slot_configs:
            cfg = self.slot_configs[key]
            cid = cfg["id"]
            
            # Extract latent vector from center region
            hidden_dim = flat_features.shape[-1]
            spatial_grid = flat_features.reshape(self.num_patches_per_side, self.num_patches_per_side, hidden_dim)
            center_slice = spatial_grid[4:10, 4:10, :].reshape(-1, hidden_dim)
            captured_vector = center_slice.mean(dim=0, keepdim=True)
            
            # Save to class dictionary memory layout
            self.registry[cid] = {
                "vector": captured_vector,
                "label": cfg["label"],
                "color": cfg["color"],
                "kalman": self.build_new_kalman_filter(),
                "initialized": False
            }
            print(f"💾 Registered target into slot memory: [{cfg['label']}]")
            
            # Trigger full action calibration if registering the main User profile (Slot 1)
            if cid == 1:
                self.calibrated_cx = None
                self.calibrated_area = None
                self.calibration_cx_accumulator = []
                self.calibration_area_accumulator = []
                self.calibration_frames_gathered = 0
                self.is_calibrating = True
                print("⏳ User registered. Calibrating action baseline coordinates...")

        # 3. Process Every Registered Target Vector Separately
        for cid, target_data in self.registry.items():
            target_vector = target_data["vector"]
            
            # Gated Similarity
            norm_features = flat_features / (flat_features.norm(dim=-1, keepdim=True) + 1e-8)
            norm_target = target_vector / (target_vector.norm(dim=-1, keepdim=True) + 1e-8)
            similarity = torch.mm(norm_features, norm_target.T).squeeze(-1).cpu().numpy()
            normalized_sim = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
            
            # Modulate explicitly using the neural probe confidence map
            gated_activation = normalized_sim * np.exp(target_log_probs)

            # High-Precision Percentile Thresholding
            adaptive_threshold = np.percentile(gated_activation, 97)
            segmentation_mask = np.zeros((self.num_patches_per_side, self.num_patches_per_side), dtype=np.uint8)
            flat_mask = segmentation_mask.flatten()
            flat_mask[gated_activation >= adaptive_threshold] = 1
            segmentation_mask = flat_mask.reshape(self.num_patches_per_side, self.num_patches_per_side)

            upscaled_mask = cv2.resize(segmentation_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            target_pixels = np.where(upscaled_mask == 1)

            kf = target_data["kalman"]

            if target_pixels[0].size > 30:
                raw_y_min, raw_y_max = np.min(target_pixels[0]), np.max(target_pixels[0])
                raw_x_min, raw_x_max = np.min(target_pixels[1]), np.max(target_pixels[1])

                raw_cx = float(raw_x_min + raw_x_max) / 2.0
                raw_cy = float(raw_y_min + raw_y_max) / 2.0
                raw_w = float(raw_x_max - raw_x_min)
                raw_h = float(raw_y_max - raw_y_min)

                kf.predict()
                if not target_data["initialized"]:
                    kf.statePost = np.array([[raw_cx], [raw_cy], [raw_w], [raw_h]], dtype=np.float32)
                    target_data["initialized"] = True
                
                measurement = np.array([[raw_cx], [raw_cy], [raw_w], [raw_h]], dtype=np.float32)
                corrected_state = kf.correct(measurement)
                fit_cx, fit_cy, fit_w, fit_h = corrected_state.flatten()

                current_area = fit_w * fit_h

                # 4. Handle Calibration Routines exclusively for Class 1 (User)
                if cid == 1 and self.is_calibrating:
                    self.calibration_cx_accumulator.append(fit_cx)
                    self.calibration_area_accumulator.append(current_area)
                    self.calibration_frames_gathered += 1
                    if self.calibration_frames_gathered >= 15:
                        self.calibrated_cx = np.mean(self.calibration_cx_accumulator)
                        self.calibrated_area = np.mean(self.calibration_area_accumulator)
                        self.is_calibrating = False
                        print(f"📊 User Calibration Complete: X={int(self.calibrated_cx)}, Area={int(self.calibrated_area)}")

                # 5. Handle UI Action Display Wording
                display_label = target_data["label"]
                box_color = target_data["color"]

                if cid == 1 and not self.is_calibrating and self.calibrated_cx is not None:
                    if current_time - self.last_action_time > self.cooldown_period:
                        if fit_cx < (self.calibrated_cx - (orig_w * 0.10)):
                            print("⬅️ ACTION TRIGGERED: [SHIFTED LEFT]")
                            self.last_action_time = current_time
                            display_label = "USER: SHIFTED LEFT"
                        elif fit_cx > (self.calibrated_cx + (orig_w * 0.10)):
                            print("➡️ ACTION TRIGGERED: [SHIFTED RIGHT]")
                            self.last_action_time = current_time
                            display_label = "USER: SHIFTED RIGHT"
                        elif current_area > (self.calibrated_area * 1.50):
                            print("🚀 ACTION TRIGGERED: [LEANED IN CLOSE]")
                            self.last_action_time = current_time
                            display_label = "USER: LEANED IN"

                # Convert to pixels and draw the bounding box
                x_min = max(0, int(fit_cx - (fit_w / 2.0) - 10))
                y_min = max(0, int(fit_cy - (fit_h / 2.0) - 10))
                x_max = min(orig_w, int(fit_cx + (fit_w / 2.0) + 10))
                y_max = min(orig_h, int(fit_cy + (fit_h / 2.0) + 10))

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, display_label, (x_min + 10, y_min + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            else:
                kf.predict()

        # If nothing is registered in memory yet, output instruction prompt overlay
        if not self.registry:
            cv2.putText(frame, "MEM EMPTY. PRESS '1' FOR USER, '2' or '3' FOR OBJECTS", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def start_camera_stream(self, stream_index=0):
        cap = cv2.VideoCapture(stream_index)
        if not cap.isOpened(): return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed_frame = self.process_and_track(frame)
            cv2.imshow("Micro-V-JEPA 2.1 Visual Core", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = LiveVJEPATracker()
    tracker.start_camera_stream(stream_index=0)