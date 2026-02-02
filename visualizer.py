from flask import Flask, render_template, Response
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
import threading, time, json, numpy as np

app = Flask(
    __name__, 
    template_folder="visualizer/templates",
    static_folder="visualizer/static",
)

TARGET_FPS = 60.0

state_lock = threading.Lock()
app_state = {
    "step_index": 0,
    "actual_fps": 0.0,
}

@app.route("/")
def index():
    return render_template("index.html")

def step_once():
    """Execute one simulation step and return positions and activations."""
    global simulator, robot_idx, max_steps, n_masses_cached, n_springs_cached
    
    t = app_state["step_index"]
    
    # Check if we need to reset
    if t >= max_steps:
        simulator.reinitialize_robots()
        app_state["step_index"] = 0
        t = 0
    
    # Execute one simulation step
    simulator.compute_com(t)
    simulator.nn1(t)
    simulator.nn2(t)
    simulator.apply_spring_force(t)
    simulator.advance(t + 1)
    
    # Extract data for the robot we're visualizing (use cached values)
    positions = simulator.x.to_numpy()[robot_idx, t + 1, :n_masses_cached]
    activations = simulator.act.to_numpy()[robot_idx, t, :n_springs_cached]
    center_of_mass = positions.mean(axis=0)
    
    app_state["step_index"] = t + 1
    
    return positions, activations, center_of_mass

@app.route("/stream")
def stream():
    """Server-sent events stream for real-time visualization."""
    global n_masses_cached, n_springs_cached
    
    def event_stream():
        # Send initial topology
        topology = {
            "type": "topology",
            "springs": robot["springs"].tolist(),
            "n_masses": int(n_masses_cached),
            "n_springs": int(n_springs_cached),
        }
        yield f"data: {json.dumps(topology)}\n\n"
        
        # Timing variables
        fps_samples = []
        last_fps_update = time.perf_counter()
        
        while True:
            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS
            
            # Step and get data
            positions, activations, center_of_mass = step_once()
            
            # Send update
            payload = {
                "type": "step",
                "positions": positions.tolist(),
                "activations": activations.tolist(),
                "center_of_mass": center_of_mass.tolist(),
                "step": app_state["step_index"],
                "fps": app_state["actual_fps"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            
            # Calculate how long this frame took (work only, not including sleep)
            frame_end = time.perf_counter()
            work_time = frame_end - frame_start
            
            # Sleep to maintain target FPS
            sleep_time = target_interval - work_time
            if sleep_time > 0.001:  # Only sleep if meaningful (>1ms)
                time.sleep(sleep_time)
            
            # Calculate actual FPS (including sleep)
            total_frame_time = time.perf_counter() - frame_start
            if total_frame_time > 0:
                fps_samples.append(1.0 / total_frame_time)
            
            # Update FPS display periodically
            current_time = time.perf_counter()
            if current_time - last_fps_update >= 0.5:
                if fps_samples:
                    with state_lock:
                        app_state["actual_fps"] = sum(fps_samples) / len(fps_samples)
                    fps_samples = []
                    last_fps_update = current_time
    
    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="robot_0.npy", help="Path to saved robot .npy file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Load robot
    print(f"Loading robot from {args.input}...")
    robot = np.load(args.input, allow_pickle=True).item()
    print(f"Robot: {robot['n_masses']} masses, {robot['n_springs']} springs")
    
    # Load config
    config = load_config(args.config)
    
    # Set up simulator config for single robot visualization
    # Use the max dimensions from training if available, otherwise use robot's actual size
    if "max_n_masses" in robot and "max_n_springs" in robot:
        config["simulator"]["n_masses"] = int(robot["max_n_masses"])
        config["simulator"]["n_springs"] = int(robot["max_n_springs"])
        print(f"Using training dimensions: max_masses={robot['max_n_masses']}, max_springs={robot['max_n_springs']}")
    else:
        # Fallback for older saved robots without max dimensions
        config["simulator"]["n_masses"] = int(robot["n_masses"])
        config["simulator"]["n_springs"] = int(robot["n_springs"])
        print("Warning: max dimensions not found in robot file, using robot's actual size (may cause issues)")
    config["simulator"]["n_sims"] = 1
    
    # Create simulator
    print("Initializing simulator...")
    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=False,
    )
    
    # Initialize with robot geometry
    simulator.initialize([robot["masses"]], [robot["springs"]])
    
    # Load control parameters if available
    if "control_params" in robot:
        simulator.set_control_params([0], [robot["control_params"]])
        print("Loaded trained control parameters")
    else:
        print("No control parameters found - using random initialization")
    
    robot_idx = 0
    
    # Cache constant values to avoid CUDA context issues
    max_steps = simulator.steps[None]
    n_masses_cached = simulator.n_masses[robot_idx]
    n_springs_cached = simulator.n_springs[robot_idx]
    
    print(f"\nVisualizer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=False, use_reloader=False)