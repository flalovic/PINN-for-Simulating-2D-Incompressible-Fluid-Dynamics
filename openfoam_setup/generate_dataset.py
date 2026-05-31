import yaml
import argparse
import subprocess

import pandas as pd
import pyvista as pv

from pathlib import Path

# Direktorij za pohranu podataka
output_dir = Path("../data/raw")

# Direktorij do config fajlova za simulacije
config_dir = Path("../config/sampling")

#Direktorij do OpenFOAM podataka
foam_dir = Path("../openfoam_setup/cavity")

def parse_args() -> dict[str]:
    parser = argparse.ArgumentParser(description="Unos fajlova za uzorkovanje fluida, kao i naziv izlazne csv datoteke")
    parser.add_argument("-cf", "--config", help="Unos fajla za uzorkovanje")
    parser.add_argument("-o", "--output", help="Unesite naziv output fajla")

    args = parser.parse_args()
    return {'config' : args.config, 'output' : args.output};

def fill_template(template_path, output_path, **kwargs):
    with open(template_path, 'r') as f:
        template = f.read()
    
    filled_content = template
    for key, value in kwargs.items():
        filled_content = filled_content.replace(f"{{{key}}}", str(value))


    with open(output_path, 'w') as f:
        f.write(filled_content)

def generate(config_file : str, output_name : str):
    with open(config_dir/config_file) as f:
        config = yaml.safe_load(f)

    dom = config['domain']
    phys = config['physics']
    sys = config['system']

    print(f"Pokrećem generisanje...")
    print(f"Brzina poklopca: {phys['boundary_conditions']['lid_velocity'][0]} m/s")
    print(f"Viskoznost (nu): {phys['kinematic_viscosity_nu']}")
    print(f"Mesh rezolucija: {dom['nx_cells']} x {dom['ny_cells']}")
    
    # 3. Ubacivanje YAML vrednosti u OpenFOAM fajlove
    # - Brzina
    fill_template(
        foam_dir / "0/U.template", 
        foam_dir / "0/U", 
        U_X=phys['boundary_conditions']['lid_velocity'][0],
        U_Y=phys['boundary_conditions']['lid_velocity'][1],
        U_Z=phys['boundary_conditions']['lid_velocity'][2]
    )

    # - Viskoznost
    fill_template(
        foam_dir / "constant/transportProperties.template", 
        foam_dir / "constant/transportProperties", 
        NU_VALUE=phys['kinematic_viscosity_nu']
    )

    # - Mreža (Mesh)
    fill_template(
        foam_dir / "system/blockMeshDict.template", 
        foam_dir / "system/blockMeshDict", 
        X_MIN=dom['x_min'], X_MAX=dom['x_max'],
        Y_MIN=dom['y_min'], Y_MAX=dom['y_max'],
        NX=dom['nx_cells'], NY=dom['ny_cells'],
        G_X = dom['mesh_grading'][0], G_Y = dom['mesh_grading'][1],
        G_Z = dom['mesh_grading'][2]
    )

    # - controlDict
    fill_template(
        foam_dir / "system/controlDict.template", 
        foam_dir / "system/controlDict", 
        END_T=sys['time_control']['end_time'],
        DELTA_T=sys['time_control']['delta_t'],
        WRITE_INT = sys['time_control']['write_interval']
    )


    print("\n[1/5] Brišem stare rezultate...")
    subprocess.run(["foamCleanTutorials"], cwd=foam_dir, capture_output=True)

    print("[2/5] Generišem mrežu (blockMesh)...")
    res_mesh = subprocess.run(["blockMesh"], cwd=foam_dir, capture_output=True, text=True)
    if res_mesh.returncode != 0:
        print(f"Greška u blockMesh:\n{res_mesh.stderr}\n{res_mesh.stdout}")
        return

    print("[3/5] Pokrećem simulaciju (icoFoam)...")
    res_foam = subprocess.run(["icoFoam"], cwd=foam_dir, capture_output=True, text=True)
    if res_foam.returncode != 0:
        print(f"Greška u icoFoam:\n{res_foam.stderr}\n{res_foam.stdout}")
        return

    print("[4/5] Ekstrahujem 2D CSV podatke na z=0.05 pomoću PyVista...")
    
    steps = ['0']
    for fold in foam_dir.iterdir():
        if fold.is_dir() and '.' in fold.name: 
            print(f" - Detektovan korak: {fold.name}")
            steps.append(fold.name)


    # 1. Kreiramo dummy fajl koji PyVista zahteva
    foam_file = foam_dir / "case.foam"
    foam_file.touch()

    # 2. Učitavamo OpenFOAM podatke za poslednji korak (0.5)
    reader = pv.OpenFOAMReader(str(foam_file))
    
    dfs = []
    for step in steps:
        reader.set_active_time_value(float(step))
        mesh = reader.read()
        fluid = mesh["internalMesh"]

        # 3. Pravimo presek na z=0.05
        slice_z = fluid.slice(normal='z', origin=(0, 0, 0.001))
        
        centers = slice_z.cell_centers()

        # 4. Pakujemo u Pandas DataFrame
        df = pd.DataFrame({
            'time' : [float(step)] * len(centers.points),
            'x': centers.points[:, 0],
            'y': centers.points[:, 1],
            'U_x': fluid['U'][:, 0],
            'U_y': fluid['U'][:, 1],
            'p': fluid['p']
        })
        dfs.append(df)

    # 5. Čuvamo kao čisti CSV na finalnoj destinaciji
    csv_dest = output_dir / f"{output_name}.csv"
    pd.concat(dfs, ignore_index=True).sort_values(by='time').to_csv(csv_dest, index=False)
    print(f"[5/5] Uspešno sačuvano u {csv_dest}!")

if __name__ == "__main__":
    ls = parse_args()
    generate(ls['config'],ls['output'])

   
