import os
import sys
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from bitstring import BitArray

from LZ77 import LZ77


class Encoder():

    def __init__(self, file_path):
        ##### Verify if file is text or image.
        self.text_file = True if os.path.splitext(file_path)[-1] == '.txt' else False

        ##### If images, get dimensions.
        if not self.text_file:
            self.dimensions = np.array(Image.open(file_path)).shape

        ##### Open and read file
        orig_file = open(file_path, "rb")
        self.sequence = orig_file.read()
        orig_file.close()


    def encode_sequence(self, search_buffer_size, look_ahead_buffer_size, second_encoding_step=False):
        ##### Instantiate LZ77 Encoder.
        self.LZ77 = LZ77(search_buffer_size, look_ahead_buffer_size)
        self.LZ77.read_sequence(np.frombuffer(self.sequence, dtype=np.uint8))

        ##### Verify if a second encoding step is required.
        if second_encoding_step:
            ##### In this case, only the triples are required. LZ77 object does not need to write a bitstring. 
            self.triples_LZ77 = self.LZ77.generate_triples()
            self.__encode_with_AE()
        else:
            ##### Only file details need to be added to the bitstring.
            self.bitstring = self.LZ77.encode_sequence()

        ##### Insert encoder header
        self.__write_encoder_header()


    ########## Private Methods

    def __encode_with_AE(self):
        # TODO: Implement.
        return


    def __write_encoder_header(self):
        ##### Bit indicating if is image or text.
        encoder_header = '0b0' if self.text_file else '0b1'

        ##### Include headers for image.
        if not self.text_file:
            # Number of channels
            three_channel_image = '1' if (len(self.dimensions) == 3) and (self.dimensions[-1] == 3) else '0'
            encoder_header += three_channel_image

            # Width and height difference
            height, width = self.dimensions[:2]
            dim_diff = width - height
            encoder_header += f', int:14={dim_diff}'

        ##### 
        self.bitstring.prepend(encoder_header)


if __name__ == "__main__":
    ##### Receives file to be compressed from command line.
    parser = argparse.ArgumentParser(description="Receives file to be encoded and binary filepath.")
    
    parser.add_argument('--file_to_compress', required=True, help='Path to file to be compressed.')
    parser.add_argument('--binary_file_path', required=False, help="Path to save binary file. "
                                                                   "If folders do not exist, they'll be created.")
    parser.add_argument('--search_buffer_length', default=32, type=int, help='Buffer size with the already encoded symbols.')
    parser.add_argument('--look_ahead_buffer_length', default=16, type=int, help='Buffer size with symbols to be encoded.')
    parser.add_argument('--second_encoding_step', action='store_true', help='Flag to set a second encoding step.')

    ##### Read command line
    args = parser.parse_args(sys.argv[1:])
    
    ##### Define directory path.
    if args.binary_file_path:
        directory = Path(os.path.dirname(args.binary_file_path))
    else:
        directory = Path("binary_files")
        file_name = os.path.splitext(os.path.basename(args.file_to_compress))[0]
        args.binary_file_path = os.path.join(directory, file_name + '.bin') 
    
    ##### Create directory.
    if not directory.exists():
        directory.mkdir(parents=True)

    ##### Encode source.
    encoder = Encoder(args.file_to_compress)
    encoder.encode_sequence(args.search_buffer_length, args.look_ahead_buffer_length, args.second_encoding_step)