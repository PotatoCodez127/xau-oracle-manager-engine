from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

def export_tfevents_to_csv(log_dir_path):
    print(f"Loading TensorBoard events from: {log_dir_path}...")
    
    # Load the event accumulator
    event_acc = EventAccumulator(log_dir_path)
    event_acc.Reload()

    # Find all the scalar metrics logged by Stable-Baselines3
    tags = event_acc.Tags().get('scalars', [])
    
    if not tags:
        print("No scalar metrics found in this log.")
        return

    # Create a directory to hold the exported CSVs
    os.makedirs("exported_metrics", exist_ok=True)

    # Loop through every metric and save it as a CSV
    for tag in tags:
        events = event_acc.Scalars(tag)
        # Extract the training step and the actual value
        data = [{"Step": e.step, "Value": e.value} for e in events]
        df = pd.DataFrame(data)
        
        # Clean the tag name so it can be saved as a valid filename 
        # (e.g., 'rollout/ep_rew_mean' becomes 'rollout_ep_rew_mean.csv')
        safe_filename = tag.replace("/", "_") + ".csv"
        save_path = os.path.join("exported_metrics", safe_filename)
        
        df.to_csv(save_path, index=False)
        print(f"Exported: {save_path}")

if __name__ == "__main__":
    # Point this directly to the folder containing your .tfevents file
    # Based on your previous logs, this should be the correct path:
    log_folder = "./models/sac_manager_tensorboard_london/SAC_1"
    export_tfevents_to_csv(log_folder)