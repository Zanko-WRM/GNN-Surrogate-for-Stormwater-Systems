import os
import re
import pandas as pd
import datetime, timedelta
import time
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import swmmtoolbox.swmmtoolbox as swm

from datetime import datetime, timedelta
from pyswmm import Simulation, Nodes, Subcatchments, Links

SWMM_EXECUTABLE_PATH= r"C:/Program Files (x86)/EPA SWMM 5.1.015/swmm5.exe"

print(f"SWMM Executable Path: {SWMM_EXECUTABLE_PATH}")

load_dotenv()
swmm_executable_path = Path("SWMM_EXECUTABLE_PATH")



def get_lines_from_textfile(path):
    with open(path, "r") as fh:
        lines_from_file = fh.readlines()
    return lines_from_file



# Utility function to read lines from a text file
def get_lines_from_textfile(path):
    with open(path, "r") as fh:
        lines_from_file = fh.readlines()
    return lines_from_file

# Function to update dates and paths in the .inp file
def update_inp_file(inp_path, start_date, start_time, end_date, end_time, dat_file_path):
    """
    Updates the .inp file with start/end dates, times, and rainfall .dat file paths.
    """
    inp = get_lines_from_textfile(inp_path)
    for i, line in enumerate(inp):
        if "START_DATE" in line:
            inp[i] = f"START_DATE           {start_date}\n"
        elif "END_DATE" in line:
            inp[i] = f"END_DATE             {end_date}\n"
        elif "START_TIME" in line:
            inp[i] = f"START_TIME           {start_time}\n"
        elif "END_TIME" in line:
            inp[i] = f"END_TIME             {end_time}\n"
        elif "REPORT_START_DATE" in line:
            inp[i] = f"REPORT_START_DATE    {start_date}\n"
        elif "REPORT_START_TIME" in line:
            inp[i] = f"REPORT_START_TIME    {start_time}\n"
        elif "FILE" in line and "*" in line:
            inp[i] = f'R1              INTENSITY 0:10     1.0      FILE       "{dat_file_path}" R1         MM\n'
    return inp

# Function to run the SWMM simulation
def run_SWMM(base_inp_path, rainfall_data_directory, simulations_path, padding_hours=0):
    """
    Runs SWMM simulations for each .dat file in rainfall_data_directory.

    Parameters:
    - base_inp_path: Path to the base .inp file
    - rainfall_data_directory: Directory containing .dat files for rainfall data
    - simulations_path: Directory to save each simulation's output
    - padding_hours: Additional hours to add to the end time of each simulation
    """
    list_of_rain_datfiles = os.listdir(rainfall_data_directory)
    columns = ["event_name", "start_date", "end_date", "end_time", "simulation_time"]
    df_info = pd.DataFrame(columns=columns)

    for event in list_of_rain_datfiles:
        print(f"Running simulation for {event}")
        rain_event_path = rainfall_data_directory / event

        # Extract start and end dates/times from the .dat file
        dat_lines = get_lines_from_textfile(rain_event_path)
        start_line = dat_lines[0].split("\t")
        start_date = f"{start_line[2]}/{start_line[3]}/{start_line[1]}"
        start_time = f"{start_line[4]}:{start_line[5]}:00"

        end_line = dat_lines[-1].split("\t")
        end_date = f"{end_line[2]}/{end_line[3]}/{end_line[1]}"
        end_time = f"{end_line[4]}:{end_line[5]}:00"

        # Add padding to end time
        end_datetime = datetime.strptime(f"{end_date} {end_time}", "%m/%d/%Y %H:%M:%S")
        end_datetime += timedelta(hours=padding_hours)
        end_date, end_time = end_datetime.strftime("%m/%d/%Y"), end_datetime.strftime("%H:%M:%S")

        # Update .inp file content
        updated_inp_content = update_inp_file(base_inp_path, start_date, start_time, end_date, end_time, rain_event_path)

        # Define simulation directory
        simulation_folder = simulations_path / event.replace(".dat", "")
        simulation_folder.mkdir(parents=True, exist_ok=True)

        # Save the updated .inp file
        inp_path = simulation_folder / "model.inp"
        with open(inp_path, "w") as fh:
            fh.writelines(updated_inp_content)

        # Save the .dat file in the simulation folder
        with open(simulation_folder / event, "w") as fh:
            fh.writelines(dat_lines)

        # Run the SWMM simulation
        rpt_file = simulation_folder / "model.rpt"
        out_file = simulation_folder / "model.out"
        start_time = time.time()
        subprocess.run([str(SWMM_EXECUTABLE_PATH), str(inp_path), str(rpt_file), str(out_file)])
        simulation_time = time.time() - start_time

        # Record simulation details
        df_event = pd.DataFrame([[event, start_date, end_date, end_time, simulation_time]], columns=columns)
        df_info = pd.concat([df_info, df_event])

    # Save execution times to an Excel file
    execution_times_path = simulations_path / "execution_times.xlsx"
    df_info.to_excel(execution_times_path, sheet_name="Execution times")
    print(f"Execution times saved to {execution_times_path}")


def extract_features_from_inp(inp_path, output_path, output_file_name):
    """
    Extracts features for subcatchments, junctions, outfalls, and conduits (including xsection data)
    from a SWMM .inp file and saves all features in a single text file.

    Parameters:
    - inp_path (Path): Path to the SWMM .inp file.
    - output_path (Path): Directory where the output file will be saved.
    - output_file_name (str): Name of the output file (e.g., "all_features.txt").

    Returns:
    - None
    """
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine the output path and file name
    output_file = output_path / output_file_name

    # Initialize data containers for each feature type
    subcatchments = []
    subareas = {}
    polygons = {}
    coordinates = {}
    junctions = []
    outfalls = []
    conduits = []
    xsections = {}

    # Flags for section detection
    reading_subcatchments = False
    reading_subareas = False
    reading_polygons = False
    reading_coordinates = False
    reading_junctions = False
    reading_outfalls = False
    reading_conduits = False
    reading_xsections = False

    # Open and read the .inp file
    with inp_path.open("r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("["):
                # Set flags based on the section headers
                reading_subcatchments = line.startswith("[SUBCATCHMENTS]")
                reading_subareas = line.startswith("[SUBAREAS]")
                reading_polygons = line.startswith("[Polygons]")
                reading_coordinates = line.startswith("[COORDINATES]")
                reading_junctions = line.startswith("[JUNCTIONS]")
                reading_outfalls = line.startswith("[OUTFALLS]")
                reading_conduits = line.startswith("[CONDUITS]")
                reading_xsections = line.startswith("[XSECTIONS]")
                continue

            # Process the [SUBCATCHMENTS] section
            if reading_subcatchments and line and not line.startswith(";;"):
                parts = line.split()
                subcatchments.append({
                    "Name": parts[0],
                    "Rain_Gage": parts[1],
                    "Outlet": parts[2],
                    "Area": float(parts[3]),
                    "Perc_Imperv": float(parts[4]),
                    "Width": float(parts[5]),
                    "Slope": float(parts[6]),
                    "CurbLen": float(parts[7]) if len(parts) > 7 else 0.0,
                    "SnowPack": parts[8] if len(parts) > 8 else None
                })

            # Process the [SUBAREAS] section
            if reading_subareas and line and not line.startswith(";;"):
                parts = line.split()
                subareas[parts[0]] = {
                    "N_Imperv": float(parts[1]),
                    "N_Perv": float(parts[2]),
                    "S_Imperv": float(parts[3]),
                    "S_Perv": float(parts[4]),
                    "PctZero": float(parts[5]),
                    "RouteTo": parts[6],
                    "PctRouted": float(parts[7]) if len(parts) > 7 else None
                }

            # Process the [Polygons] section for subcatchment boundary points
            if reading_polygons and line and not line.startswith(";;"):
                parts = line.split()
                name = parts[0]
                x, y = float(parts[1]), float(parts[2])
                if name not in polygons:
                    polygons[name] = []
                polygons[name].append((x, y))

            # Process the [COORDINATES] section for junction and outfall coordinates
            if reading_coordinates and line and not line.startswith(";;"):
                parts = line.split()
                coordinates[parts[0]] = (float(parts[1]), float(parts[2]))

            # Process the [JUNCTIONS] section
            if reading_junctions and line and not line.startswith(";;"):
                parts = line.split()
                junctions.append({
                    "Name": parts[0],
                    "Elevation": float(parts[1]),
                    "MaxDepth": float(parts[2]),
                    "InitDepth": float(parts[3]),
                    "SurDepth": float(parts[4]),
                    "Aponded": float(parts[5]) if len(parts) > 5 else 0.0,
                    "X": None,  # Placeholder for X-coordinates
                    "Y": None
                })

            # Process the [OUTFALLS] section
            if reading_outfalls and line and not line.startswith(";;"):
                parts = line.split()
                outfalls.append({
                    "Name": parts[0],
                    "Elevation": float(parts[1]),
                    "Type": parts[2],
                    "StageData": parts[3] if len(parts) > 3 else None,
                    "Gated": parts[4] if len(parts) > 4 else None,
                    "RouteTo": parts[5] if len(parts) > 5 else None,
                    "X": None,  # Placeholder for X-coordinates
                    "Y": None
                })

            # Process the [CONDUITS] section
            if reading_conduits and line and not line.startswith(";;"):
                parts = line.split()
                conduits.append({
                    "Name": parts[0],
                    "From_Node": parts[1],
                    "To_Node": parts[2],
                    "Length": float(parts[3]),
                    "Roughness": float(parts[4]),
                    "InOffset": float(parts[5]),
                    "OutOffset": float(parts[6]),
                    "InitFlow": float(parts[7]) if len(parts) > 7 else 0.0,
                    "MaxFlow": float(parts[8]) if len(parts) > 8 else 0.0,
                    "Shape": None,  # Placeholder for shape
                    "MaxDepth": None  # Placeholder for max depth from XSECTIONS
                })

            # Process the [XSECTIONS] section and store data to merge with conduits
            if reading_xsections and line and not line.startswith(";;"):
                parts = line.split()
                xsections[parts[0]] = {
                    "Shape": parts[1],
                    "MaxDepth": float(parts[2])  # Geom1 corresponds to max depth
                }

    # Assign Shape and MaxDepth from xsections to conduits
    for conduit in conduits:
        xsection_data = xsections.get(conduit["Name"])
        if xsection_data:
            conduit["Shape"] = xsection_data["Shape"]
            conduit["MaxDepth"] = xsection_data["MaxDepth"]

    # Calculate centroid coordinates for subcatchments and add subarea data
    for subcatchment in subcatchments:
        name = subcatchment["Name"]

        # Calculate centroid for the subcatchment polygon if polygon data exists
        if name in polygons:
            x_coords, y_coords = zip(*polygons[name])
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)
            subcatchment["X"], subcatchment["Y"] = centroid_x, centroid_y
        else:
            subcatchment["X"], subcatchment["Y"] = None, None

        # Add subarea details if available
        subarea_data = subareas.get(name, {})
        subcatchment.update(subarea_data)

    # Assign coordinates to junctions
    for junction in junctions:
        coord = coordinates.get(junction["Name"], (None, None))
        junction["X"], junction["Y"] = coord

    # Assign coordinates to outfalls
    for outfall in outfalls:
        coord = coordinates.get(outfall["Name"], (None, None))
        outfall["X"], outfall["Y"] = coord

    # Write all data to a single text file
    with output_file.open("w") as file:
        file.write("[SUBCATCHMENTS]\n")
        pd.DataFrame(subcatchments).to_string(file, index=False)
        file.write("\n\n[JUNCTIONS]\n")
        pd.DataFrame(junctions).to_string(file, index=False)
        file.write("\n\n[OUTFALLS]\n")
        pd.DataFrame(outfalls).to_string(file, index=False)
        file.write("\n\n[CONDUITS]\n")
        pd.DataFrame(conduits).to_string(file, index=False)

    print(f"All features saved to {output_file}")


def extract_and_save_sections(file_path, output_path):
    """
    Extract and save data for all relevant sections from the all_features.txt file.

    Parameters:
    - file_path (str): Path to the all_features.txt file.
    - output_path (str): Directory path to save the output files.

    Returns:
    - None
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    def extract_section(file_content, section_name, required_columns, numeric_columns):
        """
        Extract a specific section from the all_features.txt content and return
        only the specified columns as a DataFrame.
        """
        # Updated regex pattern to match the section data between headers
        pattern = rf"\[{section_name}\]\s*(.*?)(?=\n\[|$)"
        match = re.search(pattern, file_content, re.DOTALL)
        if not match:
            print(f"{section_name} section not found!")
            return None

        # Extract data within the section
        section_data = match.group(1).strip().split("\n")
        # Ignore lines starting with comments (;;)
        section_data = [line for line in section_data if not line.startswith(";;")]

        if not section_data:
            print(f"{section_name} section is empty or improperly formatted.")
            return None

        # Extract headers and data
        headers = section_data[0].split()
        rows = [line.split() for line in section_data[1:]]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Keep only the required columns
        df = df[required_columns]

        # Convert specified columns to numeric
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # Feature extraction functions for each section
    def extract_subcatchments(file_content):
        return extract_section(
            file_content,
            section_name="SUBCATCHMENTS",
            required_columns=["Name", "Outlet", "Area", "Perc_Imperv", "Slope", "X", "Y", "N_Imperv", "N_Perv"],
            numeric_columns=["Area", "Perc_Imperv", "Slope", "X", "Y", "N_Imperv", "N_Perv"]
        )

    def extract_junctions(file_content):
        return extract_section(
            file_content,
            section_name="JUNCTIONS",
            required_columns=["Name", "Elevation", "MaxDepth", "InitDepth", "X", "Y"],
            numeric_columns=["Elevation", "MaxDepth", "InitDepth", "X", "Y"]
        )

    def extract_outfalls(file_content):
        return extract_section(
            file_content,
            section_name="OUTFALLS",
            required_columns=["Name", "Elevation", "Type", "X", "Y"],
            numeric_columns=["Elevation"]
        )

    def extract_conduits(file_content):
        return extract_section(
            file_content,
            section_name="CONDUITS",
            required_columns=["Name", "From_Node", "To_Node", "Length", "Roughness", "Shape", "MaxDepth"],
            numeric_columns=["Length", "Roughness", "MaxDepth"]
        )

    # Read the entire file content once
    with open(file_path, "r") as file:
        file_content = file.read()

    # Extract and save subcatchments
    subcatchments_df = extract_subcatchments(file_content)
    if subcatchments_df is not None:
        subcatchments_df.to_csv(os.path.join(output_path, "subcatchments.txt"), sep="\t", index=False)
        print("Subcatchments data saved to subcatchments.txt")

    # Extract and save junctions
    junctions_df = extract_junctions(file_content)
    if junctions_df is not None:
        junctions_df.to_csv(os.path.join(output_path, "junctions.txt"), sep="\t", index=False)
        print("Junctions data saved to junctions.txt")

    # Extract and save outfalls
    outfalls_df = extract_outfalls(file_content)
    if outfalls_df is not None:
        outfalls_df.to_csv(os.path.join(output_path, "outfalls.txt"), sep="\t", index=False)
        print("Outfalls data saved to outfalls.txt")

    # Extract and save conduits
    conduits_df = extract_conduits(file_content)
    if conduits_df is not None:
        conduits_df.to_csv(os.path.join(output_path, "conduits.txt"), sep="\t", index=False)
        print("Conduits data saved to conduits.txt")


def extract_time_series(base_simulation_dir, output_dir, junction_vars, subcatchment_vars, conduit_vars):
    """
    Extracts time series data for junctions, subcatchments, and conduits from SWMM simulations.

    Parameters:
    - base_simulation_dir: Path to the directory containing simulation subfolders.
    - output_dir: Path to the directory where extracted data will be saved.
    - junction_vars: Dictionary of variables to extract for junctions.
    - subcatchment_vars: Dictionary of variables to extract for subcatchments.
    - conduit_vars: Dictionary of variables to extract for conduits.

    Returns:
    - None
    """
    base_simulation_dir = Path(base_simulation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_report_step(inp_file):
        with open(inp_file, "r") as file:
            for line in file:
                if "REPORT_STEP" in line:
                    time_parts = line.split()[1].strip().split(":")
                    return timedelta(hours=int(time_parts[0]), minutes=int(time_parts[1]), seconds=int(time_parts[2]))
        return timedelta(seconds=1)  # Default to 1-second intervals if not specified

    def process_model(model_folder):
        swmm_inp_file = model_folder / "model.inp"
        model_name = model_folder.name

        # Get report step
        report_step = get_report_step(swmm_inp_file)

        # Process Junctions
        for var_name, var_func in junction_vars.items():
            print(f"Processing Junctions {var_name} for {model_name}...")
            data = {"Time": []}
            last_saved_time = None
            sim = Simulation(str(swmm_inp_file))
            with sim:
                for step in sim:
                    current_time = sim.current_time
                    if last_saved_time is None or current_time - last_saved_time >= report_step:
                        last_saved_time = current_time
                        data["Time"].append(current_time)
                        for node in Nodes(sim):
                            if node.nodeid not in data:
                                data[node.nodeid] = []
                            data[node.nodeid].append(var_func(node))
            df = pd.DataFrame(data)
            output_file = output_dir / f"{model_name}_junctions_{var_name.lower()}.txt"
            df.to_csv(output_file, sep="\t", index=False, float_format="%.4f")
            print(f"Saved {output_file}")

        # Process Subcatchments
        for var_name, var_func in subcatchment_vars.items():
            print(f"Processing Subcatchments {var_name} for {model_name}...")
            data = {"Time": []}
            last_saved_time = None
            sim = Simulation(str(swmm_inp_file))
            with sim:
                for step in sim:
                    current_time = sim.current_time
                    if last_saved_time is None or current_time - last_saved_time >= report_step:
                        last_saved_time = current_time
                        data["Time"].append(current_time)
                        for sub in Subcatchments(sim):
                            if sub.subcatchmentid not in data:
                                data[sub.subcatchmentid] = []
                            data[sub.subcatchmentid].append(var_func(sub))
            df = pd.DataFrame(data)
            output_file = output_dir / f"{model_name}_subcatchments_{var_name.lower()}.txt"
            df.to_csv(output_file, sep="\t", index=False, float_format="%.4f")
            print(f"Saved {output_file}")

        # Process Conduits
        for var_name, var_func in conduit_vars.items():
            print(f"Processing Conduits {var_name} for {model_name}...")
            data = {"Time": []}
            last_saved_time = None
            sim = Simulation(str(swmm_inp_file))
            with sim:
                for step in sim:
                    current_time = sim.current_time
                    if last_saved_time is None or current_time - last_saved_time >= report_step:
                        last_saved_time = current_time
                        data["Time"].append(current_time)
                        for link in Links(sim):
                            if link.is_conduit:
                                if link.linkid not in data:
                                    data[link.linkid] = []
                                data[link.linkid].append(var_func(link))
            df = pd.DataFrame(data)
            output_file = output_dir / f"{model_name}_conduits_{var_name.lower()}.txt"
            df.to_csv(output_file, sep="\t", index=False, float_format="%.4f")
            print(f"Saved {output_file}")

    # Iterate through all simulation folders
    model_folders = [folder for folder in base_simulation_dir.iterdir() if folder.is_dir()]
    for model_folder in model_folders:
        print(f"Processing model: {model_folder.name}")
        process_model(model_folder)
    print("All models processed.")