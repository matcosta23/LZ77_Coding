import os
import sys
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from decimal import getcontext
from bitstring import BitStream, ReadError

from LZ77 import LZ77

# Import Adaptative Huffman Encoder
sys.path.insert(1, "../Adaptative_Huffman_Coding")
from huffman_decoder import HuffmanDecoder


########## LZ77 Decoding Class

class Decoder():

    def __init__(self, binary_file):
        ##### Read bitstring from file.
        with open(binary_file) as bin_file:
            binary = bin_file.read()
            self.bitstring = BitStream(f'0b{binary}')


    def decode_bitstring(self):
        ##### Decode encoder header
        self.__decode_header()

        ##### Instantiate LZ77
        LZ77_decoder = LZ77()

        ##### Verify if second encoding step was performed.
        second_coding_bit = self.bitstring.read('bin:1')

        ##### Decode with AE.
        if second_coding_bit is '1':
            ##### Get triples amount
            triples_bits_amount = self.bitstring.read('uint:5')
            self.triples_amount = self.bitstring.read(f'uint:{triples_bits_amount}')

            ##### Decode offsets and lengths with Adaptative binary tree.
            offsets = self.__decode_with_HC()
            match_lengths = self.__decode_with_HC()
            codes = self.__decode_with_HC()
            
            ##### Merge info and create triples
            triples = np.column_stack((offsets, match_lengths, codes))

            ##### Provide triples to LZ77 decoder.
            LZ77_decoder.read_triples(triples)
            self.sequence = LZ77_decoder.decode_sequence_from_triples()

        ##### Decode with LZ77
        else:
            self.sequence = LZ77_decoder.decode_sequence_from_bitstring(self.bitstring)

        return


    def save_decoded_file(self, decoded_file_path):
        ##### For text files, the bytes will be transformed into chars.
        if self.text_file:
            text = ''.join([chr(byte) for byte in self.sequence])
            ##### Save text in destiny path.
            decoded_file_path += '.txt'
            with open(decoded_file_path, "w") as decoded_file:
                decoded_file.write(text)
                decoded_file.close()
        
        ##### For images, the dimensions should be first obtained.
        else:
            bytes_amount = len(self.sequence)
            second_degree_coeff = [1, np.abs(self.width_height_diff), -bytes_amount]
            height = int(np.around(np.roots(second_degree_coeff).max(), 0))
            width = height + self.width_height_diff
            # Define dimensions
            self.image_dimensions = [height, width]
            self.image_dimensions.append(3) if self.three_channel_image else None
            ##### Reshape Image
            channels = 3 if self.three_channel_image else 1
            img = np.squeeze(np.array(self.sequence, dtype=np.uint8).reshape((height, width, channels)))
            ##### Include image extension
            if self.three_channel_image:
                decoded_file_path += '.png' 
                file_format = 'PNG'
            else:
                decoded_file_path += '.bmp'
                file_format = 'BMP' 
            ##### Save image
            image = Image.fromarray(img)
            image.save(decoded_file_path, format=file_format)

        return


    ##### Private Methods

    def __decode_header(self):
        ##### Verify if file is an image or text
        file_type = self.bitstring.read('bin:1')
        self.text_file = True if file_type == '0' else False

        ##### Get dimensions for image file
        if self.text_file is False:
            # Verify number of channels
            channels_amount = self.bitstring.read('bin:1')
            self.three_channel_image = True if channels_amount == '1' else False
            # Obtain width and height difference
            self.width_height_diff = self.bitstring.read('int:14')

        return


    def __decode_with_HC(self):
        ##### Get amount of bits in bitstring.
        bits_to_read = self.bitstring.read('uint:5')
        bits_amount = self.bitstring.read(f'uint:{bits_to_read}')

        ##### Read bitstring
        bitstring = self.bitstring.read(f'bin:{bits_amount}')

        ##### Decode bitstring
        huffman_decoder = HuffmanDecoder(symbols_amount=self.triples_amount)
        huffman_decoder.read_bitstream(bitstring)
        huffman_decoder.decode_with_adaptative_hc(verbose=False)
        decoded_bytes = huffman_decoder.get_decoded_bytes()

        return decoded_bytes



########## Auxiliary Methods

def menage_decoded_file_path(args):
    ##### Define directory path.
    if args.decoded_file_path:
        directory = Path(os.path.dirname(args.decoded_file_path))
    else:
        directory = Path("decoded_files")
        file_name = os.path.splitext(os.path.basename(args.binary_file_path))[0]
        args.decoded_file_path = os.path.join(directory, file_name) 
    
    ##### Create directory.
    if not directory.exists():
        directory.mkdir(parents=True)



if __name__ == "__main__":
    ##### Receives binary to be decoded from command line.
    parser = argparse.ArgumentParser(description="Receives binary file and path to save reconstructed file.")
    
    parser.add_argument('--binary_file_path', required=True, help='Path to binary file.')
    parser.add_argument('--decoded_file_path', required=False, help="Path to save decoded file. "
                                                   "If folders do not exist, they'll be created.")

    ##### Read command line
    args = parser.parse_args(sys.argv[1:])
    
    ##### Menage decoded file path
    menage_decoded_file_path(args)

    ##### Decode binary
    decoder = Decoder(args.binary_file_path)
    decoder.decode_bitstring()
    decoder.save_decoded_file(args.decoded_file_path)