import jax
import jax.numpy as jnp
import haiku as hk
import optax
import logging
import clinvar_parser as cp
from sklearn.model_selection import train_test_split
from flax import serialization
from alphamissense.data.pipeline_missense import (
    make_msa_features, 
    run_msa_tool, 
    variant_sequence_features, 
    make_variant_masked_msa, 
    read_from_gcloud,
    get_atom_positions,
    get_empty_atom_positions,
    make_msa_profile
)
from alphamissense.model.modules_missense import (
    AlphaMissense, 
    LogitDiffPathogenicityHead, 
    clipped_sigmoid_cross_entropy
)

# Assuming that AlphaMissense is a Haiku module
def forward_pass(x, is_training=True):
    model = AlphaMissense(is_training=is_training)
    return model(x)

# Transform the forward function into a pure function
transformed_model = hk.transform(forward_pass)

# Define the loss function
def loss_fn(params, inputs, targets):
    logits = transformed_model.apply(params, inputs)
    return clipped_sigmoid_cross_entropy(logits, targets)

# JAX update function
@jax.jit
def update(params, opt_state, inputs, targets):
    grads = jax.grad(loss_fn)(params, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Training hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Initialize the optimizer
optimizer = optax.adam(learning_rate)

# Data loading and preprocessing
# This is a simplified view, in practice you would use the actual data pipeline functions

# Path to the ClinVar XML file
clinvar_xml_path = './ClinVarFullRelease_00-latest.xml'

# Parse and preprocess the ClinVar dataset
variant_df = cp.parse_clinvar_xml(clinvar_xml_path)
preprocessed_df = cp.preprocess_variant_data(variant_df)
X = preprocessed_df.drop('Label',axis = 1)
Y = preprocessed_df['Label']
train_data, test_data, train_label, test_label = train_test_split(X,Y, test_size=0.2, random_state=42)
val_data, test_data, val_label, test_label = train_test_split(test_data,test_label, test_size=0.5,random_state=42)
#train_data, train_labels = read_from_gcloud('training_data')
variant_features = variant_sequence_features(train_data)
msa_features = make_msa_features(run_msa_tool(train_data))
train_dataset = make_variant_masked_msa(variant_features, msa_features)

# Initialize the model parameters
rng = jax.random.PRNGKey(42)
sample_input = jnp.array(next(iter(train_dataset))[0])  # Placeholder for a batch of input data
params = transformed_model.init(rng, sample_input)

# Initialize the optimizer state
opt_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    step = 0
    for inputs, targets in zip(train_dataset, train_labels):
        params, opt_state = update(params, opt_state, inputs, targets)

        # Logging here...
        # Log metrics to console
        training_loss = loss_fn(params, inputs, targets)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Step {step+1}/{batch_size}, Loss: {training_loss}')

# Validation and testing would follow here

# Saving the trained model parameters
# with open('trained_model.pkl', 'wb') as f:
#     pickle.dump(params, f)

# Convert JAX arrays to CPU for serialization
params_cpu = jax.device_get(params)
opt_state_cpu = jax.device_get(opt_state)

# Serialize parameters and optimizer state
serialized_params = serialization.to_bytes(params_cpu)
serialized_opt_state = serialization.to_bytes(opt_state_cpu)

# Create a directory to save the model
model_dir = './Saved_Models'
os.makedirs(model_dir, exist_ok=True)

# Save serialized parameters and optimizer state to disk
with open(os.path.join(model_dir, 'model_params.flax'), 'wb') as f:
    f.write(serialized_params)

with open(os.path.join(model_dir, 'optimizer_state.flax'), 'wb') as f:
    f.write(serialized_opt_state)

# To confirm saving process
print(f'Model parameters and optimizer state saved in {model_dir}')

