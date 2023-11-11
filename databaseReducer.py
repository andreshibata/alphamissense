import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def reduce_clinvar_xml_size(file_name, reduction_factor=10):
    # Construct the full file path
    file_path = os.path.join(os.getcwd(), file_name)

    # Get the total file size for progress estimation
    total_file_size = os.path.getsize(file_path)

    # Initialize the progress bar
    pbar = tqdm(total=total_file_size, desc='Processing ClinVarSets')

    # Count the total number of ClinVarSet elements
    total_clinvar_sets = sum(1 for event, elem in ET.iterparse(file_path, events=('end',)) if elem.tag.endswith('ClinVarSet'))

    # Calculate the number of ClinVarSets to retain
    number_to_retain = total_clinvar_sets // reduction_factor

    # Reset the iterator for actual parsing
    context = ET.iterparse(file_path, events=('start', 'end'))
    _, root = next(context)  # get root element

    removed_count = 0
    for event, elem in context:
        if event == 'end' and elem.tag.endswith('ClinVarSet'):
            if removed_count < (total_clinvar_sets - number_to_retain):
                root.remove(elem)
                removed_count += 1
            # Clear the processed elements from memory
            elem.clear()

        # Update the progress bar based on the file's current position
        pbar.update(os.path.getsize(file_path) - pbar.n)

    pbar.close()

    # Write the reduced XML back to a new file
    reduced_file_name = f'reduced_{file_name}'
    tree = ET.ElementTree(root)
    tree.write(os.path.join(os.getcwd(), reduced_file_name))

    print(f'Reduced file written to {reduced_file_name}')
