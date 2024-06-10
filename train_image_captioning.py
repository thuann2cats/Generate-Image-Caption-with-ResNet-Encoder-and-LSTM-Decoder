import torch
import torch.nn as nn
from torchvision import transforms
import sys, os, math
sys.path.append('/home/nguyenthuan49/image-captioning-project/cocoapi/PythonAPI')  # put the path to the "PythonAPI" folder here
from pycocotools.coco import COCO
from data_loader import get_loader
from models import EncoderCNN, DecoderRNN
import argparse

import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def get_arguments():
  """
  Parses command-line arguments for training script.
  """
  
  def validate_learning_rate(value):
    """Custom validation function to check for float <= 2.0."""
    value = float(value)
    if value > 2.0:
        raise argparse.ArgumentError(f"{value} is not a valid learning rate (must be <= 2.0).")
    return value
  
  parser = argparse.ArgumentParser(description="Train image captioning model.")
  # Required argument
  parser.add_argument("--data_path", type=str, required=True,
                      help="Path to the training data - required.")
  parser.add_argument("--folder_path", type=str, required=True,
                      help="Path to the folder for saving model checkpoints and logs (required).")
  
  # Optional arguments with type checks and value ranges
  parser.add_argument("--lr", type=validate_learning_rate, default=0.005,
                      help="Learning rate for the optimizer (must be <= 2.0).")
  parser.add_argument("--batch_size", type=int, default=128, metavar="8-2048", choices=range(8, 2049),
                    help="Batch size for training (default: %(default)s, range: 8-2048)")


  parser.add_argument("--embed_size", type=int, default=256, choices=range(64, 2049), metavar="64-2048",
                    help="Dimensionality of image and word embeddings (default: %(default)s, range: 64-2048)")
  parser.add_argument("--decoder_lstm_hidden_size", type=int, default=512, choices=range(64, 2049), metavar="64-2048",
                    help="Number of features in hidden state of LSTM decoder (default: %(default)s, range: 64-2048)")
  parser.add_argument("--num_epochs", type=int, default=3,
                    help="Number of training epochs (default: %(default)s)")
  parser.add_argument("--log_file", type=str, default="training_log.txt",
                    help="Name of file with saved training loss and perplexity within 'folder_path' (default: %(default)s)")
  parser.add_argument("--save_every_n_epochs", type=int, default=1,
                    help="Frequency of saving model weights (default: %(default)s)")
  parser.add_argument("--print_every_n_batches", type=int, default=100,
                    help="Frequency of printing a loss value (default: %(default)s)")
  
  parser.add_argument("--vocab_threshold", type=int, default=5, choices=range(3, 21), metavar="3-20",
                    help="Threshold for creating a vocabulary from file. If a word's frequency is under this threshold, the word would be tokenized into <unk>. This argument is ignored if '--vocab_from_file' is True(default: %(default)s, range: 3-20)")
  parser.add_argument("--vocab_from_file", action="store_true", default=False,
                    help="Load vocabulary from existing vocab file (If True, '--vocab_file_path' must be specified; if False, then new vocabulary file will be created as 'vocab.pkl'.) (default: %(default)s)")
  parser.add_argument("--vocab_file_path", type=str, help="Path to the pre-created vocabulary (.pkl) file. Required if --vocab-from-file is True")
  
  parser.add_argument("--optimizer_func", type=str, default="adam", choices=["adam", "rmsprop", "sgd"],  # Restrict choices
                        help="Optimizer function (adam, rmsprop or sgd).")
  
  parser.add_argument("--encoder_folder_name", type=str, default="saved_encoders",
                        help="Folder name for saving encoder checkpoints (automatically saved inside 'folder_path') (default: %(default)s).")
  parser.add_argument("--decoder_folder_name", type=str, default="saved_decoders",
                      help="Folder name for saving decoder checkpoints (automatically saved inside 'folder_path') (default: %(default)s).")
  parser.add_argument("--tensorboard_folder_name", type=str, default="tensorboard_logging",
                      help="Folder name for TensorBoard logging (automatically saved inside 'folder_path') (default: %(default)s).")
    
  parser.add_argument("--load_encoder_checkpoint_file", type=str,
                  help="Path to a pre-trained encoder checkpoint file (optional).")
  parser.add_argument("--load_decoder_checkpoint_file", type=str,
                      help="Path to a pre-trained decoder checkpoint file (optional).")

  return parser.parse_args()


def save_hyperparams(variable_dict: dict, folder_path: str, filename="config.json"):
    """Saving the hyparameters of the encoder and the decoder"""
    
    # Open the file for writing in JSON format
    with open(os.path.join(folder_path, filename), 'w') as file:
        import json
        json.dump(variable_dict, file, indent=4)  # Indent for readability


def train_image_captioning(
  encoder: EncoderCNN,
  decoder: DecoderRNN,
  data_loader,
  folder_path,
  device,
  num_epochs: int,
  lr=0.005,
  optimizer_func=torch.optim.Adam,
  save_every_n_epochs=1,
  print_every_n_batches=100,
  log_file_name="training_log.txt",
  tensorboard_folder="tensorboard_logging",
  saved_encoder_subfolder="saved_encoders",
  saved_decoder_subfolder="saved_decoders",
  saved_encoder_starting_idx=1,
  saved_decoder_starting_idx=1,
):
  criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

  params = list(decoder.parameters()) + list(encoder.embed.parameters())
  optimizer = optimizer_func(params=params, lr=lr)

  # num_batches = len(data_loader)
  num_batches = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
  
  encoder.to(device), decoder.to(device)
  encoder.train(), decoder.train()
  f = open(os.path.join(folder_path, log_file_name), 'w')
  tensorboard_writer = SummaryWriter(os.path.join(folder_path, tensorboard_folder))
  decoder_folder_path = os.path.join(folder_path, saved_decoder_subfolder)
  encoder_folder_path = os.path.join(folder_path, saved_encoder_subfolder)
  next_encoder_idx = saved_encoder_starting_idx
  next_decoder_idx = saved_decoder_starting_idx


  for epoch in range(1, num_epochs+1):
      for batch_idx in range(1, num_batches+1):
          # Randomly sample a caption length, and sample indices with that length
          indices = data_loader.dataset.get_train_indices()
          # Create and assign a batch sampler to retrieve a batch with the sampled indices
          new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
          data_loader.batch_sampler.sampler = new_sampler
          
          # Obtain the batch
          images, captions = next(iter(data_loader))
          
          # Move batch of images and captions to GPU if CUDA is available
          images = images.to(device)
          captions = captions.to(device)
          
          # Zero the gradients
          decoder.zero_grad()
          encoder.zero_grad()
          
          # Pass the inputs through the CNN-RNN model
          features = encoder(images)
          outputs = decoder(features, captions)
          
          # Calculate the batch loss
          loss = criterion(outputs.view(-1, decoder.vocab_size), captions.view(-1))
          loss.backward()
          optimizer.step()
          
          # Get training stats
          stats = f"Epoch [{epoch:3}/{num_epochs:3}], Step [{batch_idx:4}/{num_batches:4}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
          num_batches_so_far = (epoch - 1) * num_batches + batch_idx
          tensorboard_writer.add_scalar(tag="Train/Loss", scalar_value=loss.item(), global_step=num_batches_so_far)
          tensorboard_writer.add_scalar(tag="Train/Perplexity", scalar_value=np.exp(loss.item()), global_step=num_batches_so_far)
          
          # Track gradients
          for name, param in decoder.named_parameters():
              if param.requires_grad:
                  tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.mean(), num_batches_so_far)
          
          # Print training stats (on same line)
          print('\r' + stats, end="")
          sys.stdout.flush()
          
          # Print training stats (on different line)
          if batch_idx % print_every_n_batches == 0:
              print('\r' + stats)
              # Print training staistics to file.
              f.write(stats + '\n')
              f.flush()
              
      # Save the weights
      if epoch % save_every_n_epochs == 0:
          torch.save(decoder.state_dict(), os.path.join(decoder_folder_path, f"decoder-{next_decoder_idx}.pkl"))
          next_decoder_idx += 1
          torch.save(encoder.state_dict(), os.path.join(encoder_folder_path, f"encoder-{next_encoder_idx}.pkl"))
          next_encoder_idx += 1
          
  tensorboard_writer.close()        
  f.close()
          
          

  

def main():
  
  args = get_arguments()
  if args.vocab_from_file:
    assert args.vocab_file_path is not None
  data_path = args.data_path
  folder_path = args.folder_path
  lr = args.lr  
  batch_size = args.batch_size
  embed_size = args.embed_size
  decoder_lstm_hidden_size = args.decoder_lstm_hidden_size
  num_epochs = args.num_epochs
  log_file = args.log_file
  save_every_n_epochs = args.save_every_n_epochs
  print_every_n_batches = args.print_every_n_batches
  vocab_threshold = args.vocab_threshold
  vocab_from_file = args.vocab_from_file
  vocab_file_path = args.vocab_file_path
  if args.optimizer_func.lower() == "adam":
      optimizer_func = torch.optim.Adam
  elif args.optimizer_func.lower() == "rmsprop":
      optimizer_func = torch.optim.RMSprop
  elif args.optimizer_func.lower() == "sgd":
      optimizer_func = torch.optim.SGD
  load_encoder_checkpoint_file = args.load_encoder_checkpoint_file
  load_decoder_checkpoint_file = args.load_decoder_checkpoint_file
  encoder_folder_name = args.encoder_folder_name
  decoder_folder_name = args.decoder_folder_name
  tensorboard_folder_name = args.tensorboard_folder_name
  
  # save these variables into a file
  variable_dict = {
    "data_path": args.data_path,
    "folder_path": args.folder_path,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "embed_size": args.embed_size,
    "decoder_lstm_hidden_size": args.decoder_lstm_hidden_size,
    "num_epochs": args.num_epochs,
    "log_file": args.log_file,
    "save_every_n_epochs": args.save_every_n_epochs,
    "print_every_n_batches": args.print_every_n_batches,
    "vocab_threshold": args.vocab_threshold,
    "vocab_from_file": args.vocab_from_file,
    "vocab_file_path": args.vocab_file_path,
    "optimizer_func": args.optimizer_func,
    "load_encoder_checkpoint_file": args.load_encoder_checkpoint_file,
    "load_decoder_checkpoint_file": args.load_decoder_checkpoint_file,
    "encoder_folder_name": args.encoder_folder_name,
    "decoder_folder_name": args.decoder_folder_name,
    "tensorboard_folder_name": args.tensorboard_folder_name,
  }
  
  print("Arguments used for training:")
  print(args)
  print("\n")
  
  # Create the folder if it doesn't exist (optional)
  import os
  os.makedirs(folder_path, exist_ok=True)  # Creates folder if it doesn't exist

  save_hyperparams(variable_dict, folder_path)
  
  ### Get the loader for the training data

  transform_train = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),
                          (0.229, 0.224, 0.225)),
  ])
  
  if not vocab_from_file:
    vocab_file_path = os.path.join(folder_path, "vocab.pkl")

  data_loader = get_loader(transform=transform_train,
                          mode='train',
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_from_file=vocab_from_file,
                          vocab_file=vocab_file_path,
                          cocoapi_loc=data_path)

  vocab_size = len(data_loader.dataset.vocab)
  
  
  ### Initialize and load the previous encoder and decoder (if specified)

  # initialize encoder and decoder
  encoder = EncoderCNN(embed_size)
  decoder = DecoderRNN(embed_size, decoder_lstm_hidden_size, vocab_size)
  
  # load encoder and decoder checkpoints (if specified)
  for file, module in ((load_encoder_checkpoint_file, encoder), (load_decoder_checkpoint_file, decoder)):
      if file:
          module_state_dict = torch.load(file)
          module.load_state_dict(module_state_dict)
          
  ### Create the folders to save encoder and decoder later

  # Combine path components for subfolder
  encoder_folder_path = os.path.join(folder_path, encoder_folder_name)
  decoder_folder_path = os.path.join(folder_path, decoder_folder_name)

  # Create the subfolder if it doesn't exist
  try:
      os.makedirs(encoder_folder_path, exist_ok=True)
      os.makedirs(decoder_folder_path, exist_ok=True)
  except OSError as e:
      print(f"Error creating folder: {e}")
      
  ### Initialize other components of training

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")
  normal_repr = torch.Tensor.__repr__
  torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"  # so that VS code tensors show shapes in debugging
  
  ### Training
  
  train_image_captioning(
    encoder=encoder,
    decoder=decoder,
    data_loader=data_loader,
    folder_path=folder_path,
    device=device,
    num_epochs=num_epochs,
    lr=lr,
    optimizer_func=optimizer_func,
    save_every_n_epochs=save_every_n_epochs,
    print_every_n_batches=print_every_n_batches,
    log_file_name=log_file,
    tensorboard_folder=tensorboard_folder_name,
    saved_encoder_subfolder=encoder_folder_name,
    saved_decoder_subfolder=decoder_folder_name,
  )

        
if __name__ == "__main__":
  main()
  
