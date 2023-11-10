import xml.etree.ElementTree as ET
import pandas as pd

def parse_clinvar_xml(clinvar_xml_path):
    # Parse the XML file
    tree = ET.parse(clinvar_xml_path)
    root = tree.getroot()

    # Define namespaces to simplify finding elements
    ns = {'clinvar': 'http://www.ncbi.nlm.nih.gov/clinvar/xml'}

    # Extract relevant data from the XML
    variant_data = []
    for variant in root.findall('.//clinvar:ReferenceClinVarAssertion', ns):
        # Extract the attributes of interest, for example:
        rcv_accession = variant.find('.//clinvar:RCVAccession', ns).attrib.get('Acc')
        clinical_significance = variant.find('.//clinvar:ClinicalSignificance', ns).find('.//clinvar:Description', ns).text
        # ... more attributes as needed

        # Add to the list
        variant_data.append({
            #'GeneID': gene_id,
            'RCVAccession': rcv_accession,
            'ClinicalSignificance': clinical_significance,
            #'ClinSigSimple': clinsig_simple,
            #'HGNC_ID': hgnc_id,
            # ... more attributes as needed
        })

    # Convert to a DataFrame
    variant_df = pd.DataFrame(variant_data)
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