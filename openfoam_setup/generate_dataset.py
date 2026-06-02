import yaml
import argparse
import subprocess

import pandas as pd
import numpy as np
import pyvista as pv

from pathlib import Path

np.random.seed(42)

# Direktorij za pohranu podataka
output_dir = Path("../data")

# Direktorij do config fajlova za simulacije
config_dir = Path("../config/sampling")

template_dir = Path("../openfoam_setup/templates")

#Direktorij do OpenFOAM podataka
foam_dir = Path("../openfoam_setup/cavity")



def parse_args() -> dict[str]:
    parser = argparse.ArgumentParser(description="Unos fajlova za uzorkovanje fluida, opsega Reynoldsovog broja i izlaznih fajlova")

    parser.add_argument("-tr", "--train-config", required=True, help="Konfiguracioni fajl za trening skup")
    parser.add_argument("-te", "--test-config", required=True, help="Konfiguracioni fajl za test skup")

    parser.add_argument("-o", "--output", required=True, help="Naziv izlaznih fajlova")

    parser.add_argument("-re", "--re-range", nargs=2, type=float, metavar=("RE_MIN", "RE_MAX"), required=True, help="Opseg Reynoldsovog broja")
    parser.add_argument("--n-re", type=int, required=True, help="Broj Reynoldsovih brojeva za uzorkovanje")
    parser.add_argument("--valid-split", type=float, default=0.1, help="Udio validacionog skupa")
    parser.add_argument("--test-split", type=float, default=0.2, help="Udio test skupa")

    args = parser.parse_args()
    return {'train_config' : args.train_config, 
            'test_config' : args.test_config,
            'output' : args.output,

            're_range' : (float(args.re_range[0]), float(args.re_range[1])),
            'n_re_samples' : args.n_re,
            'valid_ratio' : args.valid_split,
            'test_ratio' : args.test_split
    }

def fill_template(template_path, output_path, **kwargs):
    with open(template_path, 'r') as f:
        template = f.read()
    
    filled_content = template
    for key, value in kwargs.items():
        filled_content = filled_content.replace(f"{{{key}}}", str(value))


    with open(output_path, 'w') as f:
        f.write(filled_content)

def generate(config_file : str, re : float) -> pd.DataFrame:
    with open(config_dir/config_file) as f:
        config = yaml.safe_load(f)

    dom = config['domain']
    phys = config['physics']
    sys = config['system']

    print(f"Pokrećem generisanje...")
    print(f"Brzina poklopca: {phys['boundary_conditions']['lid_velocity'][0]} m/s")
    print(f"Rejnoldsov broj: {re}")
    print(f"Mesh rezolucija: {dom['nx_cells']} x {dom['ny_cells']}")
    
    # 3. Ubacivanje YAML vrednosti u OpenFOAM fajlove
    # - Brzina
    fill_template(
        template_dir / "U.template", 
        foam_dir / "0/U", 
        U_X=phys['boundary_conditions']['lid_velocity'][0],
        U_Y=phys['boundary_conditions']['lid_velocity'][1],
        U_Z=phys['boundary_conditions']['lid_velocity'][2]
    )

    # - Viskoznost
    U = phys['boundary_conditions']['lid_velocity'][0]
    L = dom['x_max'] - dom['x_min']

    nu = U * L / re

    fill_template(
        template_dir / "physicalProperties.template", 
        foam_dir / "constant/physicalProperties", 
        NU_VALUE=nu
    )

    # - Mreža (Mesh)
    fill_template(
        template_dir / "blockMeshDict.template", 
        foam_dir / "system/blockMeshDict", 
        X_MIN=dom['x_min'], X_MAX=dom['x_max'],
        Y_MIN=dom['y_min'], Y_MAX=dom['y_max'],
        NX=dom['nx_cells'], NY=dom['ny_cells'],
        G_X = dom['mesh_grading'][0], G_Y = dom['mesh_grading'][1],
        G_Z = dom['mesh_grading'][2]
    )

    # - controlDict
    fill_template(
        template_dir / "controlDict.template", 
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
            're' : [re] * len(centers.points),
            'x': centers.points[:, 0],
            'y': centers.points[:, 1],
            'U_x': fluid['U'][:, 0],
            'U_y': fluid['U'][:, 1],
            'p': fluid['p']
        })
        dfs.append(df)

    # 5. Vraćamo generisani DataFrame za zadato re
    print(f"[5/5] Funkcija generate(re = {re}) je uspješno izvršena.")
    return pd.concat(dfs, ignore_index=True).sort_values(by='time')

def generate_train_valid_test(args : dict):
    re_min, re_max = args['re_range']
    n_re_samples = args['n_re_samples']
    valid_split_index = int(n_re_samples * args['valid_ratio'])
    test_split_index = valid_split_index + int(n_re_samples * args['test_ratio'])

    re_values = np.linspace(re_min, re_max, n_re_samples)
    np.random.shuffle(re_values)

    train_dfs = []
    valid_dfs = []
    test_dfs = []

    for i, re in enumerate(re_values):
        if i < valid_split_index:
            valid_dfs.append(
                generate(args['train_config'], re)
            )
        elif i < test_split_index:
            test_dfs.append(
                generate(args['test_config'], re)
            )
        else:
            train_dfs.append(
                generate(args['train_config'], re)
            )

    pd.concat(train_dfs, ignore_index=True).\
        sort_values(by='re', kind='stable').to_csv(output_dir / f"{args['output']}_train.csv", index=False)
    pd.concat(valid_dfs, ignore_index=True).\
        sort_values(by='re', kind='stable').to_csv(output_dir / f"{args['output']}_valid.csv", index=False)
    pd.concat(test_dfs, ignore_index=True).\
        sort_values(by='re', kind='stable').to_csv(output_dir / f"{args['output']}_test.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    generate_train_valid_test(args)

# python generate_dataset.py -tr train.yaml -te test.yaml -o data -re 100 1000 --n-re 50 --valid-split 0.2 --test-split 0.2

