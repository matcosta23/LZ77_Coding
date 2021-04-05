import os
import sys
import math
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from decimal import getcontext
from bitstring import BitArray

import pyae
from LZ77 import LZ77

# Import Adaptative Huffman Encoder
sys.path.insert(1, "../Adaptative_Huffman_Coding")
from huffman_encoder import HuffmanEncoder


class Encoder():

    def __init__(self, file_path):
        ##### Verify if file is text or image.
        self.text_file = True if os.path.splitext(file_path)[-1] == '.txt' else False
        ##### Open and read text file
        if self.text_file:
            orig_file = open(file_path, "rb")
            self.sequence = orig_file.read()
            orig_file.close()
        ##### Open and read image
        else:
            image_array = np.array(Image.open(file_path))
            self.dimensions = image_array.shape
            self.sequence = image_array.flatten()



    def encode_sequence(self, search_buffer_size, look_ahead_buffer_size, second_encoding_step=False):
        ##### Instantiate LZ77 Encoder.
        self.LZ77 = LZ77()
        self.LZ77.create_buffers(search_buffer_size, look_ahead_buffer_size)
        self.LZ77.read_sequence(np.frombuffer(self.sequence, dtype=np.uint8))

        ##### Verify if a second encoding step is required.
        if second_encoding_step:
            ##### In this case, only the triples are required. LZ77 object does not need to write a bitstring. 
            self.triples_LZ77 = np.array(self.LZ77.generate_triples())
            self.__encode_with_Adaptative_HC()
            # Signaling for second coding
            self.bitstring.prepend('0b1')
        else:
            ##### Only file details need to be added to the bitstring.
            self.bitstring = self.LZ77.encode_sequence()
            self.bitstring.prepend('0b0')

        ##### Insert encoder header
        self.__write_encoder_header()


    def save_binary_file(self, binary_file_path):
        with open(binary_file_path, "wb") as bin_file:
            bin_file.write(self.bitstring.bin.encode())
            bin_file.close()


    ########## Private Methods

    def __encode_with_Adaptative_HC(self):
        # NOTE: Only offsets and match_lengths will be further encoded
        offsets = self.triples_LZ77[:, 0]
        match_lengths = self.triples_LZ77[:, 1]
    
        ##### Generate bitstrings
        offset_bs = self.__generate_Adaptative_HC_bitstrings(offsets)
        length_bs = self.__generate_Adaptative_HC_bitstrings(match_lengths)

        ##### Get total amount of triples.
        triples_amount = self.triples_LZ77.shape[0]
        bits_to_write_triples_amount = len(bin(triples_amount)[2:])

        ##### Create bitstring
        self.bitstring = BitArray(f'uint:5={bits_to_write_triples_amount}, uint:{bits_to_write_triples_amount}={triples_amount}')

        ##### Write offsets and match lengths bitstrings in the main bitstring.
        self.__write_bitstring_in_main_bitstring(offset_bs)
        self.__write_bitstring_in_main_bitstring(length_bs)
        
        ##### Write codes in the bitstring.
        codes = self.triples_LZ77[:, 2]
        for code in codes:
            self.bitstring.append(f'uint:8={code}')

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

        ##### Write header
        self.bitstring.prepend(encoder_header)


    def __generate_Adaptative_HC_bitstrings(self, sequence):
        ##### Get symbols amount
        symbols_amount = len(np.unique(sequence))

        ##### Instantiate Huffman Encoder
        huffman_encoder = HuffmanEncoder(symbols_amount=symbols_amount)
        huffman_encoder.read_sequence_array(sequence)

        ##### Generate bitstrings
        huffman_encoder.instantiate_bitstream()
        huffman_encoder.encode_with_adaptative_hc()
        bitstring = huffman_encoder.get_binary_string()

        return bitstring


    def __write_bitstring_in_main_bitstring(self, bitstring):
        ##### Include the number of bits used for the bitstring.
        bits_amount = len(bitstring)
        bits_to_write_bits_amount = len(bin(bits_amount)[2:])
        self.bitstring.append(f'uint:5={bits_to_write_bits_amount}, uint:{bits_to_write_bits_amount}={bits_amount}')

        ##### Write bitstring
        self.bitstring.append(f'bin={bitstring}')

        return



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
    encoder.save_binary_file(args.binary_file_path)