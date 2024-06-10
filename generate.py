import argparse
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
sys.path.append('/home/nguyenthuan49/image-captioning-project/cocoapi/PythonAPI')  # put the path to PythonAPI folder here
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
from models import EncoderCNN, DecoderRNN
import torch

# modify the repr on torch tensor so shape is more easily visible on Debug panel
normal_repr = torch.Tensor.__repr__ 
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"  


def get_arguments():
    """
    Parses command-line arguments using argparse.

    Returns:
        A namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate captions for an image.")

    # Required arguments
    parser.add_argument("--image_path", help="Path to the image to generate captions for. (If this is specified, --output_image_folder is ignored.)")
    parser.add_argument("--coco_test_data_folder", required=True,
                        help="Path to the COCO test data folder. Even when user provides an image, this argument is still needed because the vocabulary file still needs to be loaded with a test loader")
    parser.add_argument("--output_image_folder", help="Path to the folder to save the random test image.")
    parser.add_argument("--saved_encoder_path", required=True,
                        help="Path to the saved encoder model.")
    parser.add_argument("--saved_decoder_path", required=True,
                        help="Path to the saved decoder model.")
    parser.add_argument("--vocab_file_path", required=True,
                        help="Path to the vocabulary file.")

    # Optional arguments with defaults and ranges
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam search width (default: 5, range: 1-50).")
    parser.add_argument("--max_generation_len", type=int, default=20,
                        help="Maximum generation length (default: 20, range: 5-50).")

    # Parse arguments
    args = parser.parse_args()

    # Validate optional arguments (optional)
    if args.beam_width < 1 or args.beam_width > 50:
        raise ValueError("Beam width must be between 1 and 50.")
    if args.max_generation_len < 5 or args.max_generation_len > 50:
        raise ValueError("Max generation length must be between 5 and 50.")
    
    if args.image_path is None and args.output_image_folder is None:
        raise ValueError("If generating captions on a test COCO image, --output_image_folder must be specified.")

    return args


def load_encoder_from_state_dict(encoder_path: str, device) -> EncoderCNN:
    module_state_dict = torch.load(encoder_path, map_location=device)
    embed_size = module_state_dict['embed.bias'].shape[0]
    encoder = EncoderCNN(embed_size=embed_size).to(device)
    encoder.load_state_dict(module_state_dict)
    return encoder


def load_decoder_from_state_dict(decoder_path: str, device, vocab_size=None) -> DecoderRNN:
    module_state_dict = torch.load(decoder_path, map_location=device)
    vocab_size2, embed_size = module_state_dict["word_embedding_layer.weight"].shape
    if vocab_size is not None:
        assert vocab_size == vocab_size2
    vocab_size3, hidden_size = module_state_dict["projection.weight"].shape
    assert vocab_size2 == vocab_size3
    decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size2)
    decoder.to(device)
    decoder.load_state_dict(module_state_dict)
    return decoder


def generate_tokenized_captions(
    preprocessed_image: torch.Tensor, 
    encoder: EncoderCNN, 
    decoder: DecoderRNN,
    beam_width: int = 5,
    max_generation_len: int = 20) -> list[int]:
    # Obtain the embedded image features.
    features = encoder(preprocessed_image)
    
    outputs = decoder.sample(features, output_multiple=True, beam_width=beam_width, max_len=max_generation_len)
    
    # if multiple outputs are returned
    captions = []
    for candidate in outputs:
        tokens = [token.item() for token in candidate[0]]
        captions.append(tokens)
    
    return captions
    
    
def clean_sentence(tokenized_sequence, data_loader):
    sentence = ' '.join([data_loader.dataset.vocab.idx2word[token] for token in tokenized_sequence])
    return sentence


def save_image(image: torch.Tensor, save_folder: str):
    # Visualize sample image
    plt.imshow(np.squeeze(image))
    plt.title('test image')

    from datetime import datetime
    # Get current date and time in a specific format
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Adjust format as needed (YYYY-MM-DD_HH-MM-SS)
    filename = f"image_(saved_time_{now}).png"  # Combine prefix with formatted date and time

    # Save the plot with the generated filename
    plt.savefig(os.path.join(save_folder, filename))
    
    return filename

def main():
    ### Parse the arguments
    # output: args with arguments: coco_test_data_folder, saved_encoder_path, saved_decoder_path, vocab_file_path (required)
    # optional: beam_width, default 5, range 1-50
    # max_generation_len, default 20, range 5-50
    args: argparse.Namespace
    args = get_arguments()
    
    #  Define a transform to pre-process the testing images.
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),
    ])
    
    ### Get the test loader
    # need the coco_test_data_folder and vocab_file_path argument
    test_loader: torch.utils.data.DataLoader # output: test_loader
    
    
    # Create the data loader.
    print(f"Loading the vocabulary from {args.vocab_file_path} ", end="")
    test_loader = get_loader(transform=transform_test,    
                            mode='test',
                            vocab_from_file=True,
                            vocab_file=args.vocab_file_path,
                            cocoapi_loc=args.coco_test_data_folder)
    print("DONE")
        
    if args.image_path is None:
    
        # Obtain sample image before and after pre-processing.
        print(f"Grabbing a random test image from the test loader... ", end="")
        orig_image, preprocessed_image = next(iter(test_loader))
        print("DONE")
        
    elif args.image_path:
        from PIL import Image

        # Load an image
        print(f"Processing the user-specified image: {args.image_path}... ", end="")
        img = Image.open(args.image_path)
        preprocessed_image = transform_test(img)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        print("DONE")
        
    
    ### Load the encoder and decoer
    # need: saved_encoder_path, saved_decoder_path, device
    # need to infer embed_size, hidden_size (from state dict), vocab_size (from test_loader)
    encoder: torch.nn.Module # output: encoder and decoder modules
    decoder: torch.nn.Module
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(test_loader.dataset.vocab)
    print(f"Loading saved encoder from {args.saved_encoder_path}... ", end="")
    encoder = load_encoder_from_state_dict(args.saved_encoder_path, device)
    print("DONE")
    print(f"Loading saved decoder from {args.saved_decoder_path}... ", end="")
    decoder = load_decoder_from_state_dict(args.saved_decoder_path, device, vocab_size)
    print("DONE")
    encoder.eval(), decoder.eval()
    
    ### Get the most likely token sequence
    # need: encoder, decoder, test_loader, device
    captions: list[int] # output captions: the list of tokens (each token is an int)
    
    preprocessed_image = preprocessed_image.to(device)
    
    print(f"Beam search in progress...  ", end="")
    captions = generate_tokenized_captions(preprocessed_image, encoder=encoder, decoder=decoder, beam_width=args.beam_width, max_generation_len=args.max_generation_len)
    print("DONE")
    
    ### Convert to natural sentences
    # need: captions
    # convert to a list of natural sentences
    captions_str: list[str] = [] # output captions_str
    for tokenized_seq in captions:
        sentence = clean_sentence(tokenized_seq, data_loader=test_loader)
        captions_str.append(sentence)
    
    ### Output to the console
    if args.image_path is None:
        new_image_filename = save_image(orig_image, args.output_image_folder)
        print(f"\nTest image saved to {os.path.join(args.output_image_folder, new_image_filename)}\n")
    elif args.image_path:
        print(f"\nInput image: {args.image_path}\n")

    print("Generated captions: \n")
    for i, caption in enumerate(captions_str):
        print(f"{i+1:4}. {caption}")
    

if __name__ == "__main__":
    main()




