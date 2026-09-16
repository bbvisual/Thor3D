import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thor3d import ThorRenderer
from thor3d.spec import default_agent_mode, default_fov

DEBUG_DIR = "/data/roy/Thor3D/debug_sweeps"

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    scene_name = "FloorPlan1"
    agent_mode = default_agent_mode(scene_name)
    fov = default_fov(scene_name)

    # Test rotations pointing back into the room toward the island
    test_positions = [
        {"x": -1.0, "z": -1.0, "rot": 180, "hor": 30},
        {"x": -1.0, "z": -1.0, "rot": 225, "hor": 30},
        {"x": -1.0, "z": -1.0, "rot": 270, "hor": 30},
        {"x": -1.0, "z": -1.0, "rot": 315, "hor": 30},
    ]

    with ThorRenderer(width=1280, height=1280, gpu_device=1, quality="High") as r:
        for i, pos in enumerate(test_positions):
            r.controller.reset(scene=scene_name, agentMode=agent_mode, snapToGrid=False)
            
            r.controller.step(
                action="TeleportFull",
                x=pos["x"],
                y=0.9019,
                z=pos["z"],
                rotation=dict(x=0, y=pos["rot"], z=0),
                horizon=pos["hor"],
                standing=True
            )

            spec = r.capture_spec(scene_name, agent_mode=agent_mode)
            spec.camera.field_of_view = fov
            
            res = r.render(spec)
            filename = f"x_{pos['x']}_z_{pos['z']}_rot_{pos['rot']}_hor_{pos['hor']}.png"
            Image.fromarray(res.rgb).save(os.path.join(DEBUG_DIR, filename))
            print(f"Rendered: {filename}")

if __name__ == "__main__":
    main()