import os
import sys
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from decimal import getcontext
from bitstring import BitStream, ReadError

import pyae
from LZ77 import LZ77



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
            triples_bits_amount = self.bitstring.read('uint:5')
            self.triples_amount = self.bitstring.read(f'uint:{triples_bits_amount}')

            offsets = self.__decode_with_AE()
            match_lenghts = self.__decode_with_AE()

            ##### Read codes until the end of the bitstring
            codes = []
            while True:
                try:
                    codes.append(self.bitstring.read('uint:8'))
                except ReadError:
                    break
            ##### Verify that the number of codes written is the same as offsets and match lenghts
            assert self.triples_amount == len(codes), "Number of codes is different from offsets and lengths."
            
            ##### Merge info and create triples
            triples = np.column_stack((offsets, match_lenghts, codes))

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


    def __decode_with_AE(self):
        ##### Get frequency table
        # Get maximum element
        element_bits_amount = self.bitstring.read('uint:5')
        max_element = self.bitstring.read(f'uint:{element_bits_amount}')
        # Read counts sorted by element and generate frequency table.
        counts_bits_amount = self.bitstring.read('uint:5')
        frequency_table = {}
        for element in range(max_element + 1):
            frequency_table[element] = self.bitstring.read(f'uint:{counts_bits_amount}')
        
        ##### Get words amount.
        bits_amount_flag = self.bitstring.read('bin:1')
        bits_to_read = 15 if bits_amount_flag == '0' else 20
        words_amount = self.bitstring.read(f'uint:{bits_to_read}')

        ##### Change decimal precision
        getcontext().prec = 10 * words_amount

        ##### Read bits
        ae_binary = '0.' + self.bitstring.read(f'bin:{words_amount * 16}')
        float_message = pyae.bin2float(ae_binary)

        ##### Instantiate AE and decode message
        AE = pyae.ArithmeticEncoding(frequency_table=frequency_table)
        decoded_bytes, _ = AE.decode(encoded_msg=float_message, 
                                     msg_length=self.triples_amount, probability_table=AE.probability_table)

        return decoded_bytes



if __name__ == "__main__":
    ##### Receives binary to be decoded from command line.
    parser = argparse.ArgumentParser(description="Receives binary file and path to save reconstructed file.")
    
    parser.add_argument('--binary_file', required=True, help='Path to binary file.')
    parser.add_argument('--decoded_file_path', required=False, help="Path to save decoded file. "
                                                   "If folders do not exist, they'll be created.")

    ##### Read command line
    args = parser.parse_args(sys.argv[1:])
    
    ##### Define directory path.
    if args.decoded_file_path:
        directory = Path(os.path.dirname(args.decoded_file_path))
    else:
        directory = Path("decoded_files")
        file_name = os.path.splitext(os.path.basename(args.binary_file))[0]
        args.decoded_file_path = os.path.join(directory, file_name) 
    
    ##### Create directory.
    if not directory.exists():
        directory.mkdir(parents=True)

    ##### Decode binary
    decoder = Decoder(args.binary_file)
    decoder.decode_bitstring()
    decoder.save_decoded_file(args.decoded_file_path)