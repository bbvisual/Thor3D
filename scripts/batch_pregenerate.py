import os
import sys
import json
import argparse
import itertools
import shutil
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thor3d import ThorRenderer

def main():
    parser = argparse.ArgumentParser(description="AI2-THOR Batch Combinatorial Generator")
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

    # Resolution reduced to 1024x576 to minimize pixel count and save storage space
    with ThorRenderer(width=1024, height=576, gpu_device=1, quality="Ultra") as r:
        r.controller.reset(
            scene=args.scene, 
            snapToGrid=False,
            renderInstanceSegmentation=True
        )
        
        # NOTE: Update these coordinates based on your web_explorer findings for this specific trial!
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

        # 2. Extract Native 2D Bounding Boxes
        target_types = {"Potato", "Bowl", "Bread",  "Tomato", "Egg","CellPhone" ,"Mug"}
        live_objects = event.metadata['objects']
        detections2D = event.instance_detections2D
        
        target_objects = [
            o for o in live_objects 
            if o['objectType'] in target_types and o['objectId'] in detections2D
        ]
        
        bboxes_data = []
        items = []

        for idx, obj in enumerate(target_objects):
            obj_id = obj['objectId']
            obj_type = obj['objectType']
            
            items.append((obj_type.lower(), obj_id))
            
            start_x, start_y, end_x, end_y = detections2D[obj_id]
            bboxes_data.append({
                "id": idx + 1,
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

        # 3. Combinatorial Image Generation
        total_generated = 0
        for k in range(1, len(items) + 1):
            for combo in itertools.combinations(items, k):
                # Ensure all items are enabled before processing combination
                for _, obj_id in items:
                    r.controller.step(action="EnableObject", objectId=obj_id)

                # Disable combination targets
                for _, obj_id in combo:
                    r.controller.step(action="DisableObject", objectId=obj_id)

                labels_removed = sorted([label for label, _ in combo])
                # Change combinatorial files to .jpg and apply JPEG compression
                filename = f"removed_{'_'.join(labels_removed)}.jpg"
                
                Image.fromarray(r.controller.last_event.frame).save(os.path.join(out_dir, filename), format="JPEG", quality=85)
                total_generated += 1

        for _, obj_id in items:
            r.controller.step(action="EnableObject", objectId=obj_id)

        print(f"\nCompleted {args.trial}. {total_generated} permutations saved.")

if __name__ == "__main__":
    main()