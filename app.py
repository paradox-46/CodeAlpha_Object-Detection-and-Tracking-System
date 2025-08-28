import cv2
import numpy as np
from collections import defaultdict, deque, Counter
import time

class AdvancedObjectDetector:
    def __init__(self):
        # Initialize network
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 
                                    else cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 
                                   else cv2.dnn.DNN_TARGET_CPU)
        
        # Load COCO class labels
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Tracking and analytics
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.class_counter = Counter()  # Changed from defaultdict to Counter
        self.fps_history = deque(maxlen=10)
        
        # Visualization
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Get output layers
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def update_tracking(self, boxes, class_ids):
        current_ids = set()
        updated_tracks = {}
        
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            center = (box[0] + box[2]//2, box[1] + box[3]//2)
            
            min_dist = float('inf')
            track_id = None
            
            for tid, history in self.track_history.items():
                if len(history) > 0:
                    last_pos = history[-1]
                    dist = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                    if dist < min_dist and dist < 50:
                        min_dist = dist
                        track_id = tid
            
            if track_id is None:
                track_id = len(self.track_history) + 1
            
            self.track_history[track_id].append(center)
            updated_tracks[track_id] = (box, class_id)
            current_ids.add(track_id)
            self.class_counter[class_id] += 1
        
        # Clean up old tracks
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                del self.track_history[tid]
        
        return updated_tracks
    
    def draw_analytics(self, frame, tracks):
        # Draw tracking paths
        for tid, history in self.track_history.items():
            if tid in tracks:
                color = self.colors[tracks[tid][1] % 255]
                for i in range(1, len(history)):
                    cv2.line(frame, history[i-1], history[i], color, 2)
        
        # Draw bounding boxes and labels
        for tid, (box, class_id) in tracks.items():
            x, y, w, h = box
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{self.classes[class_id]} (ID: {tid})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display analytics
        fps = np.mean(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show top 3 detected classes
        top_classes = self.class_counter.most_common(3)
        for i, (class_id, count) in enumerate(top_classes):
            cv2.putText(frame, 
                       f"{self.classes[class_id]}: {count}", 
                       (10, 60 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        return frame

def main():
    detector = AdvancedObjectDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        detector.fps_history.append(fps)
        prev_time = curr_time
        
        # Detect objects
        boxes, confidences, class_ids = detector.detect_objects(frame)
        
        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indices) > 0:
            boxes = [boxes[i] for i in indices.flatten()]
            class_ids = [class_ids[i] for i in indices.flatten()]
            
            # Update tracking
            tracks = detector.update_tracking(boxes, class_ids)
            
            # Draw results
            frame = detector.draw_analytics(frame, tracks)
        
        cv2.imshow("Advanced Object Tracking", frame)
        
        if cv2.waitKey(1) == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()