import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def reduce_clinvar_xml_size(file_name, reduction_factor=10):
    # Construct the full file path
    file_path = os.path.join(os.getcwd(), file_name)

    # Load the XML file
    print("Loading File\n")
    tree = ET.parse(file_path)
    root = tree.getroot()
    print("File Loaded\n")

    print("Starting reduction:\n")
    # ClinVar specific: Identify the main element that holds variant records
    for child in list(root):
        if child.tag.endswith('ClinVarSet'):
            total_elements = len(child)
            elements_to_keep = total_elements // reduction_factor
            elements_to_remove = total_elements - elements_to_keep

            # Initialize the progress bar
            with tqdm(total=elements_to_remove + 1, desc="Overall Progress") as pbar:
                # Remove elements to achieve the desired reduction
                for _ in range(elements_to_remove):
                    del child[-1]  # Delete the last element
                    pbar.update(1)  # Update progress bar after each element removal

                # Update progress bar before saving the file
                pbar.set_description("Saving Reduced File")
                pbar.refresh()

                # Save the modified XML
                reduced_file_path = file_path.replace('.xml', '_reduced.xml')
                tree.write(reduced_file_path)

                # Final update to progress bar
                pbar.update(1)
    print("Reduction Completed")

    return reduced_file_path

# Usage
print("Starting Program\n")
file_name = 'ClinVarFullRelease_2023-10.xml'  # Replace with your actual file name
reduced_file_path = reduce_clinvar_xml_size(file_name)
print(f'Reduced file saved at: {reduced_file_path}')