import torch

# 5234 -> Reasoning
# 6901 -> History
# 7307 -> Science
# 3707 -> Coding
# 5290 -> Math

FEATURE_ID=5234
steering_strength=0.10
sae=None
def set_sae(model):
  global sae
  sae=model

def sae_steering_hook(module,inputs,outputs):
  if sae is None:
    return outputs
  if isinstance(outputs,tuple):
    hidden=outputs[0]
  else:
    hidden=outputs

  hidden=hidden.clone()
  device=hidden.device

  # Shape:[batch, seq_len, hidden_size]
  last_token=hidden[:,-1,:]

  with torch.no_grad():
    features=sae.relu(sae.encoder(last_token.float()))
    original_features=features.clone()
    features[:,FEATURE_ID]*=(1+steering_strength)
    delta=sae.decoder(features)-sae.decoder(original_features)
  delta=delta.to(device)
  hidden[:,-1,:]+=delta

  if isinstance(outputs,tuple):
    return(hidden,)+outputs[1:]
  return hidden
