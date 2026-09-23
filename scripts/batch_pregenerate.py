import os
import sys
import json
import argparse
import shutil
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thor3d import ThorRenderer

def main():
    parser = argparse.ArgumentParser(description="AI2-THOR Asset Extractor (Base & Bounding Boxes)")
    parser.add_argument("--scene", default="FloorPlan1", help="The AI2-THOR scene to load (e.g., FloorPlan1)")
    parser.add_argument("--trial", default="Trial_1_FP1_Island", help="The output folder name (e.g., Trial_1_FP1_Island)")
    args = parser.parse_args()

    # Output to the specific trial directory
    out_dir = f"/data/roy/RoboDel/public/Prerendered_Scenes/{args.trial}"
    
    # Wipe the directory if it already exists to ensure a clean slate
    if os.path.exists(out_dir):
        print(f"Clearing existing contents in {out_dir}...")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Initializing {args.scene} -> Saving to {args.trial}")

    # Resolution set to match your live layout
    with ThorRenderer(width=1024, height=576, gpu_device=1, quality="Ultra") as r:
        r.controller.reset(
            scene=args.scene, 
            snapToGrid=False,
            renderInstanceSegmentation=True
        )
        
        # Teleport configuration
        event = r.controller.step(
            action="TeleportFull",
            x=-0.75,
            y=0.901,
            z=0.5,
            rotation=dict(x=0, y=90, z=0),
            horizon=30,
            standing=True
        )

        # 1. Save base image as JPEG with 85% quality compression
        base_path = os.path.join(out_dir, "base.jpg")
        Image.fromarray(event.frame).save(base_path, format="JPEG", quality=85)
        print(f"Saved base image: {base_path}")

        # 2. Extract Native 2D Bounding Boxes for FloorPlan1 targets
        target_types = {"CellPhone", "Egg", "Bowl", "Bread", "Tomato"}
        live_objects = event.metadata['objects']
        detections2D = event.instance_detections2D
        
        # Filter strictly by target types and ensure they have valid 2D detections
        target_objects = [
            o for o in live_objects 
            if o['objectType'] in target_types and o['objectId'] in detections2D
        ]
        
        bboxes_data = []
        items = []

        for obj in target_objects:
            obj_id = obj['objectId']
            obj_type = obj['objectType']
            
            items.append((obj_type.lower(), obj_id))
            
            start_x, start_y, end_x, end_y = detections2D[obj_id]
            bboxes_data.append({
                "id": obj_id,  # Stores the exact AI2-THOR string ID directly
                "label": obj_type,
                "x": int(start_x),
                "y": int(start_y),
                "width": int(end_x - start_x),
                "height": int(end_y - start_y),
                "isClicked": False
            })
        
        print(f"Target items detected ({len(items)}): {[label for label, _ in items]}")

        bbox_path = os.path.join(out_dir, "bounding_boxes.json")
        with open(bbox_path, 'w') as f:
            json.dump(bboxes_data, f, indent=4)
        print(f"Saved native bounding boxes to: {bbox_path}")
        print(f"\nCompleted extraction for {args.trial}.")

if __name__ == "__main__":
    main()