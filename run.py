import random
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import os
import requests
import json
import base64
import sys


def get_docking_data(ligand, protein):
    response = requests.get(f"{os.environ['API_URL']}?endpoint=run_docking&smiles={ligand}&target={protein}")
    if response.status_code == 200:
        data = response.json()
        if 'images' in data:
            for i, img in enumerate(data['images']):
                open(f'{img}.png', 'wb').write(base64.b64decode(data['images'][img]))
        if 'error' in data:
            return None
        return data
    else:
        time.sleep(60)
        return None


def run(config):
    log = open(f'logs/{config.split("/")[-1].replace(".json", ".txt")}', 'a')
    config = json.load(open(config, 'r'))
    load_dotenv()
    for protein in config['proteins']:
        log.write(f'Starting protein {protein}\n\n')
        for trial in range(config['num_trials']):
            log.write(f'Starting trial {trial+1}\n\n')
            #Reset on previous best every 25 iterations
            for i in range(config['conversation_length']//25):
                system_instructions = config['system_instructions'].replace('__PROTEIN__', protein)
                client = genai.Client(api_key=os.environ['API_KEY'])
                chat = client.chats.create(model=config['model'], config=types.GenerateContentConfig(system_instruction=system_instructions))
                prompt = [config['init_prompt'].replace('__PROTEIN__', protein)] if i==0 else [config['continued_init_prompt'].replace('__SMILES__', best_molecule).replace('__BINDING_AFFINITY__', str(random.uniform(-15, -5)) if config['randomize_affinity'] else str(best_affinity)).replace('__PROTEIN__', protein).replace('__HYDROGEN_BONDS__', str(best_hbonds))] + best_molecule_images

                best_affinity = float('inf')
                best_molecule = ""
                best_hbonds = 0
                best_molecule_images = []
                for j in range(25):
                    try:
                        response = chat.send_message(prompt)
                        log.write('-------------------------------' + '\n')
                        log.write(f'Iteration {i*25+j} Prompt: ' + prompt[0] + '\n' + '-------------------------------' + '\n')
                        log.write('Response: ' + response.text + '\n\n')
                        log.flush()
                        ligand = response.text.replace('```', '').split()[-1]
                        data = get_docking_data(ligand, protein)

                        if data and data['binding_affinity'] != 0:
                            log.write(f'Docking result: {ligand} {data["binding_affinity"]} {data["hydrogen_bonds"]} {protein}\n\n')
                            if data['binding_affinity']<best_affinity:
                                best_affinity = data['binding_affinity']
                                best_molecule = ligand
                                best_hbonds = data["hydrogen_bonds"]
                                best_molecule_images = [types.Part.from_bytes(data=data['images'][img], mime_type="image/png") for img in data['images']]

                            prompt = [config['feedback_prompt'].replace('__SMILES__', ligand).replace('__BINDING_AFFINITY__', str(random.uniform(-15, -5)) if config['randomize_affinity'] else str(data['binding_affinity'])).replace('__PROTEIN__', protein).replace('__HYDROGEN_BONDS__', str(data['hydrogen_bonds']))]
                            if config['include_images']:
                                for img in data['images']:
                                    prompt.append(types.Part.from_bytes(data=data['images'][img], mime_type="image/png"))
                        else:
                            log.write(f'Docking result: {ligand} failed {protein}\n\n')
                            prompt = [config['invalid_prompt'].replace('__PROTEIN__', protein).replace('__SMILES__', ligand)]
                    except Exception as e:
                        print(e)
                        time.sleep(60)
                        continue
                print(f"Best affinity: {best_affinity}")
run(sys.argv[1])