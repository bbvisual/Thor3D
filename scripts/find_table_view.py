import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thor3d import ThorRenderer, ObjectEdit
from thor3d.spec import default_agent_mode, default_fov

OUT_DIR = "/data/roy/RoboDel/public/Prerendered_Scenes/robothor_scene_01"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scene_name = "FloorPlan1"
    agent_mode = default_agent_mode(scene_name)
    fov = default_fov(scene_name)

    width, height = 1280, 720

    with ThorRenderer(width=width, height=height, gpu_device=1, quality="Ultra") as r:
        r.controller.reset(scene=scene_name, agentMode=agent_mode, snapToGrid=False)
        
        # Locate the kitchen island / counter object dynamically
        objects = r.objects()
        counter = next((o for o in objects if o['objectType'] == 'CounterTop'), None)
        
        if counter:
            c_pos = counter['position']
            print(f"Found CounterTop at: {c_pos}")
            
            # Place agent just outside the counter position looking inward
            r.controller.step(
                action="TeleportFull",
                x=c_pos['x'] - 1.0,
                y=0.9019,
                z=c_pos['z'],
                rotation=dict(x=0, y=90, z=0),
                horizon=30,
                standing=True
            )

        spec = r.capture_spec(scene_name, agent_mode=agent_mode)
        spec.camera.field_of_view = fov

        # Render and save base image
        base_res = r.render(spec)
        base_path = os.path.join(OUT_DIR, "base.png")
        Image.fromarray(base_res.rgb).save(base_path)
        print(f"Saved dynamic base image to {base_path}")

        # Render object variants for items on the table
        target_types = ["Apple", "Book", "Bread", "Bowl", "Tomato", "Pan", "Cup", "Lettuce"]
        target_objects = [o for o in objects if o['objectType'] in target_types]

        for idx, obj in enumerate(target_objects):
            box_id = idx + 1
            edit = ObjectEdit(object_id=obj['objectId'], remove=True)
            removed_res = r.render(spec, edit=edit)
            
            output_path = os.path.join(OUT_DIR, f"removed_box_{box_id}.png")
            Image.fromarray(removed_res.rgb).save(output_path)
            print(f"Generated: removed_box_{box_id}.png -> Removed {obj['objectType']}")

if __name__ == "__main__":
    main()