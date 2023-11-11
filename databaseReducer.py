import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def reduce_clinvar_xml_size(file_name, reduction_factor=10):
    # Construct the full file path
    file_path = os.path.join(os.getcwd(), file_name)

    # Parse the XML file and get the root
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Calculate the number of ClinVarSets to retain
    total_clinvar_sets = len(root.findall('.//ClinVarSet'))
    number_to_retain = total_clinvar_sets // reduction_factor

    # Create a list of ClinVarSet elements to be removed
    clinvar_sets_to_remove = []

    # Select ClinVarSets to be removed
    for i, clinvar_set in enumerate(root.findall('.//ClinVarSet')):
        if i % reduction_factor != 0:
            clinvar_sets_to_remove.append(clinvar_set)

    # Remove the selected ClinVarSets with a progress bar
    for clinvar_set in tqdm(clinvar_sets_to_remove, desc='Reducing file size'):
        root.remove(clinvar_set)

    # Write the reduced XML back to a new file
    reduced_file_name = f'reduced_{file_name}'
    tree.write(os.path.join(os.getcwd(), reduced_file_name))

    print(f'Reduced file written to {reduced_file_name}')
