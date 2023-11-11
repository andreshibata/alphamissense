import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def reduce_clinvar_xml_size(file_name, reduction_factor=10):
    # Construct the full file path
    file_path = os.path.join(os.getcwd(), file_name)

    # Count the total elements to determine how many to remove
    total_elements = 0
    print('Initial Parsing...\n')
    for event, elem in ET.iterparse(file_path, events=('end',)):
        if elem.tag.endswith('ClinVarSet'):
            total_elements += 1
            elem.clear()  # Clear the element to save memory
    elements_to_keep = total_elements // reduction_factor
    elements_to_remove = total_elements - elements_to_keep

    # Re-parse the file and remove elements
    print("Starting reduction:\n")
    tree = ET.ElementTree()
    root = None
    with tqdm(total=elements_to_remove, desc="Overall Progress") as pbar:
        for event, elem in ET.iterparse(file_path, events=('start', 'end')):
            if event == 'start' and root is None:
                root = elem  # Capture the root element
                tree._setroot(root)
            if event == 'end' and elem.tag.endswith('ClinVarSet'):
                if elements_to_remove > 0:
                    root.remove(elem)  # Remove the element
                    elements_to_remove -= 1
                    pbar.update(1)  # Update progress bar after each element removal
                elem.clear()  # Clear the element to save memory

    # Save the modified XML
    reduced_file_path = file_path.replace('.xml', '_reduced.xml')
    tree.write(reduced_file_path)
    print("Reduction Completed")

    return reduced_file_path

# Usage
print("Starting Program\n")
file_name = 'ClinVarFullRelease_00-latest.xml'  # Replace with your actual file name
reduced_file_path = reduce_clinvar_xml_size(file_name)
print(f'Reduced file saved at: {reduced_file_path}')