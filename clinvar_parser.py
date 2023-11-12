import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_clinvar_xml(clinvar_xml_path, reduced_size):
    # Create an empty list to store the variant data
    variant_data = []
    
    # Use iterparse to parse the file incrementally
    print("Loading Data File...\n")
    
    # Use iterparse to parse the file incrementally
    context = ET.iterparse(clinvar_xml_path, events=('end',))
    # Fast iteration using iterparse
    i = 0
    for event, elem in tqdm(context, desc="Parsing Variants"):
        if elem.tag.endswith('ReferenceClinVarAssertion'):
            # Extract the attributes of interest, for example:
            rcv_accession = elem.find('.//ClinVarAccession')
            clinical_significance = elem.find('.//ClinicalSignificance')
            # ... more attributes as needed
                
            if clinical_significance.find('.//Description') is not None:
                clinical_significance = clinical_significance.find('.//Description').text
            if rcv_accession.attrib.get('Type') != 'RCV':
                print("Error, not RCV record")
            else:
                rcv_accession = rcv_accession.attrib.get('Acc')

            # Add to the list
            variant_data.append({
                'RCVAccession': rcv_accession,
                'ClinicalSignificance': clinical_significance,
                # ... more attributes as needed
                #elem
            })

            # It's important to clear elements to free up memory
            elem.clear()
            
            i = i + 1
            #Limits the number of data points taken from database
            if i > reduced_size:
                break;

    # Convert to a DataFrame
    variant_df = pd.DataFrame(variant_data)
    print("Data File Loaded!\n")
    return variant_df

def preprocess_variant_data(variant_df):
    # Preprocess the variant data for input into the model
    # This could involve encoding categorical variables, normalizing numerical data, etc.
    # For example:
    variant_df['EncodedSignificance'] = variant_df['ClinicalSignificance'].map(significance_encoding)
    # ... more preprocessing as needed
    return variant_df

def significance_encoding(significance):
    # Map the clinical significance to an encoded value
    # This is a placeholder function - the actual encoding would depend on the model's needs
    encoding_map = {
        'Pathogenic': 1,
        'Likely pathogenic': 0.75,
        'Uncertain significance': 0.5,
        'Likely benign': 0.25,
        'Benign': 0
    }
    return encoding_map.get(significance, 0)

# Path to the ClinVar XML file
#clinvar_xml_path = 'path_to_clinvar_xml.xml'

# Parse and preprocess the ClinVar dataset
#variant_df = parse_clinvar_xml(clinvar_xml_path)
#preprocessed_df = preprocess_variant_data(variant_df)

# Now `preprocessed_df` is ready to be used as input for the model