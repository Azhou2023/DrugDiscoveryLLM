import base64
import random
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from api import get_docking_data

load_dotenv()

def inference(protein, num_iterations, information="basic"):
    system_instructions = "You are an assistant helping a lab generate ligands that can bind to certain proteins with high binding affinity."
    client = genai.Client(api_key=os.environ["API_KEY"])
    chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(system_instruction=system_instructions))

    init_prompt = f"We will collaborate on generating a ligand for the {protein} protein with high binding affinity. I will give you the output from docking software after each attempt. Make sure your response contains a SMILES string at the end, and make sure that the SMILES string is valid. Start by generating an initial ligand."

    try:
        response = chat.send_message(init_prompt)
        ligand = response.text.split()[-1]
        data = get_docking_data(ligand, protein)
        invalid_ligand = False
        if data:
            print(f"Initial generation: Molecule: {ligand} | Binding affinity: {data["binding_affinity"] if not invalid_ligand else 0}")
        else:
            print("Initial generation: Invalid ligand")
            invalid_ligand = True
    except Exception as e:
        print(str(e))
        time.sleep(60)      # wait so the RPM limit resets
        invalid_ligand = True

    affinities = [data["binding_affinity"]] if not invalid_ligand else []
    best_affinity = data["binding_affinity"] if not invalid_ligand else 0
    best_molecule = ligand if not invalid_ligand else ""
    best_iteration = 0
    start_time = time.time()
    for i in range(num_iterations):
        iterated_prompt = ""
        if not invalid_ligand:
            if information == "Basic":
                iterated_prompt = f"Software shows that the molecule you generated ({ligand}) had a binding affinity of {data["binding_affinity"]} to {protein}. Based on this information, generate a better ligand, following the same answer format"
            elif information == "Extra":
                iterated_prompt = [f"Software shows that the molecule you generated, {ligand}, had a binding affinity of {data["binding_affinity"]} to {protein} and formed {data["hydrogen_bonds"]} hydrogen bonds. Pictures of the interaction between the ligand and protein are also included, and the protein surface is colored by electrostatics. First, describe what you see in the image, and relate it to the observed binding affinity. Then discuss how to improve the binding. Based on this information, generate a better ligand, following the same answer format. Please keep trying improvements, and never say the ligand is final. Ensure the last word of your response is a valid SMILES string.",
                                   types.Part.from_bytes(data=data["images"]["view0"], mime_type="image/jpeg"), types.Part.from_bytes(data=data["images"]["view1"], mime_type="image/jpeg"), types.Part.from_bytes(data=data["images"]["view2"], mime_type="image/jpeg")]
            else:
                rand = random.uniform(-15, -5)
                iterated_prompt = f"Software shows that the molecule you generated ({ligand}) had a binding affinity of {rand} to {protein}. Based on this information, generate a better ligand, following the same answer format"
        else:
            iterated_prompt = f"Software shows that the molecule you generated ({ligand}) had a binding affinity of 0 for {protein}. Generate a new ligand, following the same answer format."
        try:
            response = chat.send_message(iterated_prompt)
            open(f'imgs/{i+1}.txt', 'w').write(response.text)
            ligand = response.text.split()[-1]

            data = get_docking_data(ligand, protein)
            if data and data["binding_affinity"]!=0:
                invalid_ligand=False
                affinities.append(data["binding_affinity"])
                if data["binding_affinity"]<best_affinity:
                    best_affinity = data["binding_affinity"]
                    best_molecule = ligand
                    best_iteration = i+1
                print(f"Iteration {i+1}: Molecule: {ligand} | Binding affinity: {data["binding_affinity"]}")
            else:
                invalid_ligand=True
                print(f"Iteration {i+1}: Invalid ligand")
        except Exception as e:
            print(str(e))
            time.sleep(60)
            continue

    print(f"Best ligand: {best_molecule} with binding affinity {best_affinity} at iteration {best_iteration}")
    if len(affinities)>=2:
        pcc = pearsonr(range(len(affinities)), affinities)[0]
        pvalue = pearsonr(range(len(affinities)), affinities)[1]
        scatter = plt.scatter(range(len(affinities)), affinities)
        plt.xlabel("Iterations")
        plt.ylabel("Binding affinity")
        ax = scatter.axes
        ax.invert_yaxis()
        plt.show()
        print(f"PCC: {pcc} | P-value: {pvalue}")

    execution_time = time.time() - start_time
    return {"best_affinity": best_affinity, "best_molecule": best_molecule,"best_iteration": best_iteration, "pcc": pcc, "pvalue": pvalue, "execution_time": execution_time}