from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Set the random seed for reproducibility
    np.random.seed(config["seed"])
    # Randomly sample robots
    # NOTE: the number of robots should match the number of parallel simulations allocated in the simulator config
    robots = load_robots(num_robots=config["simulator"]["n_sims"])

    # Extract the number of masses and springs in each robot
    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]
    # Find the largest number of masses and springs in any robot
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    # Save the maximum number of masses and springs to the simulator config
    # NOTE: this is essential to ensure the simulator allocates the correct amount of memory for the simulation
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

    # Initialize the simulator
    simulator = Simulator(sim_config=config["simulator"], taichi_config=config["taichi"],seed=config["seed"], needs_grad=True)

    # Extract the masses and springs from each robot
    masses = [robot["masses"] for robot in robots]
    springs = [robot["springs"] for robot in robots]
    # Initialize the simulator state with the unique geometries of the robots
    simulator.initialize(masses, springs)

    # Train the robots to perform locomotion
    # The number of learning steps is specified in the configuration
    fitness_history = simulator.train() # numpy array of shape (n_robots, n_learning_steps)
    # Save the fitness history to a file
    np.save("fitness_history.npy", fitness_history)

    # Select the final fitness of each robot after training
    fitness = fitness_history[:, -1]
    # Sort the robots by fitness
    ranking = np.argsort(fitness)[::-1]
    ranked_robots = [robots[i] for i in ranking]
    # Select the top 3 performers
    top_3_idxs = ranking[:3]
    top_3_robots = [robots[i] for i in top_3_idxs]
    # Extract the control parameters of the top 3 performers
    top_3_control_params = simulator.get_control_params(top_3_idxs)
    # Save each of the top 3 robots and their control parameters to a file
    for i in range(3):
        robot = top_3_robots[i]
        control_params = top_3_control_params[i]
        robot["control_params"] = control_params
        # Save the max dimensions used during training so visualizer can recreate the same memory allocation setup in the simulator
        robot["max_n_masses"] = max_num_masses
        robot["max_n_springs"] = max_num_springs
        np.save(f"robot_{i}.npy", robot)