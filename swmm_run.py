from pathlib import Path
import os
import swmm_utils as ud
from swmm_utils import extract_features_from_inp, extract_and_save_sections, extract_time_series
import swmmtoolbox.swmmtoolbox as swm

import pandas as pd
from tqdm import tqdm

# Define paths based on your directory structure
inp_path = Path(r"C:/SWS_GNN/model_data/swmm_model/base_model.inp")
rainfall_dats_directory = Path(r"C:/SWS_GNN/model_data/rainfall_data")
#rainfall_dats_directory = Path(r"C:/SWS_GNN/model_data/rainfall_data/synthetic_events")
simulations_path = Path(r"C:/SWS_GNN/model_data/simulations")
swmm_executable_path = Path("SWMM_EXECUTABLE_PATH")
output_path = Path(r"C:/SWS_GNN/model_data/export_data")


# Define specific subdirectories relative to the base output path
time_series_dir = output_path / "time_series"
constant_features_dir = output_path / "constant_features"

# Ensure the directories exist
time_series_dir.mkdir(parents=True, exist_ok=True)
constant_features_dir.mkdir(parents=True, exist_ok=True)


# Variables to extract for each feature
junction_vars = {
    "Depth": lambda node: node.depth,
    "Head": lambda node: node.head,
    "Inflow": lambda node: node.total_inflow,
    "LateralInflow": lambda node: node.lateral_inflow,
    "SurchargeDepth": lambda node: node.surcharge_depth,
    "FloodVolume": lambda node: node.flooding,
}

subcatchment_vars = {
    "Evaporation": lambda sub: sub.evaporation_loss,
    "Infiltration": lambda sub: sub.infiltration_loss,
    "Rainfall": lambda sub: sub.rainfall,
    "Runoff": lambda sub: sub.runoff,
}

conduit_vars = {
    "Depth": lambda link: link.depth,
    "Volume": lambda link: link.volume,
    "Flow": lambda link: link.flow,
}

# Workflow to run SWMM and extract data
def run_swmm_and_extract(inp_path, rainfall_dats_directory, simulations_path, constant_features_dir, time_series_dir):
    """
    Runs SWMM simulations and extracts static features and dynamic time series data.
    """
    print("Starting SWMM simulations and data extraction...")

    # Step 1: Run SWMM simulations
    ud.run_SWMM(inp_path, rainfall_dats_directory, simulations_path, padding_hours=0)

    # Step 2: Extract static features from the .inp file
    extract_features_from_inp(inp_path, constant_features_dir, "all_features.txt")

    # Step 3: Extract and save sections
    combined_features_file = constant_features_dir / "all_features.txt"
    extract_and_save_sections(combined_features_file, constant_features_dir)

    # Step 4: Extract time series data
    extract_time_series(simulations_path, time_series_dir, junction_vars, subcatchment_vars, conduit_vars)

    print("SWMM processing and data extraction completed.")

# # Run the workflow
if __name__ == "__main__":
    run_swmm_and_extract(inp_path, rainfall_dats_directory, simulations_path, constant_features_dir, time_series_dir)



# Define paths based on your directory structure
#### output_file_name = "all_features.txt"

# Run the model for each rainfall event
#ud.run_SWMM(inp_path, rainfall_dats_directory, simulations_path, padding_hours=0)

# Define the full path to all_features.txt
#file_path = output_path / output_file_name
#extract_features_from_inp(inp_path, output_path, output_file_name)
#extract_and_save_sections(file_path, output_path)

