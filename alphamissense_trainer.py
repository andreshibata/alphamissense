import jax
import jax.numpy as jnp
import haiku as hk
import optax
import logging
import clinvar_parser as cp
import tensorflow as tf
import numpy as np
import pandas as pd
from jax import grad, jit, vmap
from sklearn.model_selection import train_test_split
from flax import serialization
from alphamissense.model import config
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
    clipped_sigmoid_cross_entropy,
    RunModel
)

# Data loading and preprocessing
# This is a simplified view, in practice you would use the actual data pipeline functions

# Path to the ClinVar XML file
clinvar_xml_path = './ClinVarFullRelease_00-latest.xml'

# Parse and preprocess the ClinVar dataset
variant_df = cp.parse_clinvar_xml(clinvar_xml_path, 100000)
preprocessed_df = cp.preprocess_variant_data(variant_df)
print(preprocessed_df)
X = preprocessed_df.drop('RCVAccession',axis = 1)
Y = preprocessed_df['EncodedSignificance']
X_encoded = pd.get_dummies(X)
X_filled = X_encoded.fillna(0)
Y_encoded = pd.get_dummies(Y)
Y_filled = Y_encoded.fillna(0)
train_data, test_data, train_label, test_label = train_test_split(X_filled,Y_filled, test_size=0.2, random_state=42)
val_data, test_data, val_label, test_label = train_test_split(test_data,test_label, test_size=0.5,random_state=42)
#train_data, train_labels = read_from_gcloud('training_data')
#variant_features = variant_sequence_features(train_data)
#msa_features = make_msa_features(run_msa_tool(train_data))
#train_dataset = make_variant_masked_msa(variant_features, msa_features)

# Initialize the model parameters
rng = jax.random.PRNGKey(42)
sample_input = jnp.array(train_data.to_numpy(dtype = 'float32'))  # Placeholder for a batch of input data
sample_labels = jnp.array(train_label.to_numpy(dtype = 'float32'))
#params = transformed_model.init(rng, sample_input)

# Example debugging snippet
print("Data type:", type(sample_labels))
print("Data shape:", sample_labels.shape)  # If batch_data is an array
print(sample_labels[0])

#Initalize Model
cfg = config.model_config()
model = RunModel(cfg, params=None)
model.init_params = (sample_input,42)

# Training hyperparameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

def loss_fn(params, data, labels, random_seed, model):
    predictions = model.apply(params, jax.random.PRNGKey(random_seed), data)
    return jnp.mean((predictions - labels) ** 2)  # Example: mean squared error
    
# Gradient function
grad_fn = jit(grad(loss_fn))

# Initialize the optimizer state
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(model.params)

# Training loop
for epoch in range(num_epochs):
    num_batches = 0
    total_loss = 0
    for inputs, targets in zip(sample_input, sample_labels):
        #sample_input, opt_state = update(sample_input, opt_state, inputs, targets)

        # Logging here...
        # Log metrics to console
        #grads = jit(grad(loss_fn(model.params, input, targets, 42)))
        loss, grads = jax.value_and_grad(loss_fn)(model.params, inputs, targets, 42, model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model.params = optax.apply_updates(model.params, updates)
        
        training_batch_loss = loss_fn(params, jax.tree_map(jnp.asarray,inputs), targets, 42)
        total_loss += training_batch_loss
        num_batches +=1
    
    avg_loss = total_loss/num_batches
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}')

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

