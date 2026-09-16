import os
import sys
import io
import base64
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image

# 1. Parse command-line arguments before starting the server
parser = argparse.ArgumentParser(description="AI2-THOR Web Explorer")
# Set default to FloorPlan1 so it still works if you forget to pass an argument
parser.add_argument("scene", nargs="?", default="FloorPlan1", help="The floor plan to load (e.g., FloorPlan2)")
args = parser.parse_args()
TARGET_SCENE = args.scene

# Import your custom renderer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thor3d import ThorRenderer

renderer_instance = None
r = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global renderer_instance, r
    print(f"Initializing renderer with scene: {TARGET_SCENE}")
    renderer_instance = ThorRenderer(width=800, height=600, gpu_device=1, quality="Medium")
    r = renderer_instance.__enter__()
    # 2. Feed the parsed argument into the reset function
    r.controller.reset(scene=TARGET_SCENE)
    yield
    if renderer_instance:
        renderer_instance.__exit__(None, None, None)

app = FastAPI(lifespan=lifespan)

@app.get("/step")
def step(action: str = "Pass"):
    global r
    
    if action == "MoveAhead": r.controller.step("MoveAhead")
    elif action == "MoveBack": r.controller.step("MoveBack")
    elif action == "MoveRight": r.controller.step("MoveRight")
    elif action == "MoveLeft": r.controller.step("MoveLeft")
    elif action == "RotateRight": r.controller.step("RotateRight")
    elif action == "RotateLeft": r.controller.step("RotateLeft")
    elif action == "LookUp": r.controller.step("LookUp")
    elif action == "LookDown": r.controller.step("LookDown")
    
    event = r.controller.last_event
    agent = event.metadata['agent']
    
    img = Image.fromarray(event.frame)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "image": img_str,
        "x": round(agent['position']['x'], 4),
        "y": round(agent['position']['y'], 4),
        "z": round(agent['position']['z'], 4),
        "rotation": round(agent['rotation']['y'], 4),
        "horizon": round(agent['cameraHorizon'], 4)
    }

@app.get("/", response_class=HTMLResponse)
def index():
    # 3. Dynamically inject the scene name into the web UI title
    return f"""
    <html>
    <body style="background:#1e1e1e; color:#fff; text-align:center; font-family:monospace;">
        <h2>AI2-THOR Live Explorer: <span style="color:#FF9800;">{TARGET_SCENE}</span></h2>
        <img id="view" src="" style="width:800px; height:600px; border:2px solid #555; border-radius:8px;"/>
        
        <div style="margin-top:20px; font-size:18px; background:#2d2d2d; display:inline-block; padding:15px; border-radius:8px;">
            <b>X:</b> <span id="x" style="color:#4CAF50;">0</span> | 
            <b>Y:</b> <span id="y" style="color:#4CAF50;">0</span> | 
            <b>Z:</b> <span id="z" style="color:#4CAF50;">0</span> <br><br>
            <b>Rotation:</b> <span id="rot" style="color:#2196F3;">0</span> | 
            <b>Horizon:</b> <span id="hor" style="color:#2196F3;">0</span>
        </div>
        
        <p style="color:#aaa;">Controls: <b>W/A/S/D</b> to move | <b>Q/E</b> to rotate | <b>Up/Down Arrows</b> to pitch</p>
        
        <script>
            let isProcessing = false;
            
            async function sendAction(action) {{
                if (isProcessing) return;
                isProcessing = true;
                try {{
                    let res = await fetch('/step?action=' + action);
                    let data = await res.json();
                    document.getElementById('view').src = "data:image/jpeg;base64," + data.image;
                    document.getElementById('x').innerText = data.x;
                    document.getElementById('y').innerText = data.y;
                    document.getElementById('z').innerText = data.z;
                    document.getElementById('rot').innerText = data.rotation;
                    document.getElementById('hor').innerText = data.horizon;
                }} catch(err) {{
                    console.error(err);
                }}
                isProcessing = false;
            }}
            
            window.addEventListener('keydown', function(e) {{
                if(e.key === 'w' || e.key === 'W') sendAction('MoveAhead');
                if(e.key === 's' || e.key === 'S') sendAction('MoveBack');
                if(e.key === 'a' || e.key === 'A') sendAction('MoveLeft');
                if(e.key === 'd' || e.key === 'D') sendAction('MoveRight');
                if(e.key === 'e' || e.key === 'E') sendAction('RotateRight');
                if(e.key === 'q' || e.key === 'Q') sendAction('RotateLeft');
                if(e.key === 'ArrowUp') {{ e.preventDefault(); sendAction('LookUp'); }}
                if(e.key === 'ArrowDown') {{ e.preventDefault(); sendAction('LookDown'); }}
            }});
            
            sendAction('Pass');
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)