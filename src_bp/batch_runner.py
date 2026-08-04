import pandas as pd
import subprocess
import json
import tempfile
import os
import ast

# Load the experiments from CSV
path = '/home/avaldez/projects/GraphDeepLearning/inputs/'
dataset = 'semeval' # autext23, autext24, semeval, coling
filename = f'experiments_isg_{dataset}.csv'
csv_path = os.path.join(path, filename) 
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    config = row.dropna().to_dict()

    if isinstance(config.get("leave_out_sources"), str):
        config["leave_out_sources"] = [x.strip() for x in config["leave_out_sources"].split(",")]

    # Save temp config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    print(f" Running experiment {idx + 1} with config: {config_path}")

    try: 
        
        # Run test_isg_graph.py with config file as argument
        subprocess.run(["python", "test_isg_graph.py", "--config_path", config_path], check=True)
        
        # Mark as done if successful
        df.at[idx, '_done'] = True
        print(f"Finished experiment {idx + 1}")

    except subprocess.CalledProcessError as e:
        df.at[idx, '_done'] = 'Error'
        print(f"Failed experiment {idx + 1}: {e}")
    
    finally:
        # Optional: delete the temp config file
        os.remove(config_path)

    # Save updated CSV with the _done column updated
    df.to_csv(csv_path, index=False)
