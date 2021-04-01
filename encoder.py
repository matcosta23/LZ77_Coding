import os
import sys
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from decimal import getcontext
from bitstring import BitArray

import pyae
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
            self.triples_LZ77 = np.array(self.LZ77.generate_triples())
            self.__encode_with_AE()
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

    def __encode_with_AE(self):
        ##### Generate dictionaries.
        offsets = self.triples_LZ77[:, 0]
        match_lengths = self.triples_LZ77[:, 1]
    
        offset_dict = self.__count_values_and_create_dict(offsets)
        match_length_dict = self.__count_values_and_create_dict(match_lengths)

        ##### Encode with arithmetic encoder.
        offset_bitstring = self.__encode_from_frequency_table(offsets, offset_dict)
        match_length_bitstring = self.__encode_from_frequency_table(match_lengths, match_length_dict)

        ##### Get total amount of triples.
        triples_amount = self.triples_LZ77.shape[0]
        bits_to_write_triples_amount = len(bin(triples_amount)[2:])

        ##### Create bitstring
        self.bitstring = BitArray(f'uint:5={bits_to_write_triples_amount}, uint:{bits_to_write_triples_amount}={triples_amount}')

        ##### Write offsets and match lengths in the bitstring.
        self.__write_AE_header_and_bitstring(offset_bitstring, offset_dict)
        self.__write_AE_header_and_bitstring(match_length_bitstring, match_length_dict)
        
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

    
    def __count_values_and_create_dict(self, list):
        ##### Count Values
        elements, occurences = np.unique(list, return_counts=True)
        ##### Create dictionary
        list_dict = {}
        for element, count in zip(elements, occurences):
            list_dict[element] = count

        return list_dict

    
    def __encode_from_frequency_table(self, message_to_encode, frequency_table):
        ##### Uses the message length to define the precision.
        getcontext().prec = len(message_to_encode)
        ##### Instantiate AE
        AE = pyae.ArithmeticEncoding(frequency_table=frequency_table, save_stages=True)
        ##### Encode message
        float_message, _, interval_min_value, interval_max_value = AE.encode(msg=message_to_encode,
                                                                             probability_table=AE.probability_table)
        ##### Generate binary
        binary_code, _ = AE.encode_binary(interval_min_value, interval_max_value)

        # TODO: Remove this decoding process.
        # Get float message from binary
        decoded_float_message = pyae.bin2float(binary_code)
        # Decode message
        decoded_message, _ = AE.decode(encoded_msg=decoded_float_message,
                                       msg_length=len(message_to_encode),
                                       probability_table=AE.probability_table)

        return binary_code[2:]


    def __write_AE_header_and_bitstring(self, bitstring_AE, frequency_table):
        ##### Write frequency table
        max_element = max(frequency_table.keys())
        max_counts = max(frequency_table.values())

        element_bits_amount = len(bin(max_element)[2:])
        counts_bits_amount = len(bin(max_counts)[2:])

        self.bitstring.append(f'uint:5={element_bits_amount}')
        self.bitstring.append(f'uint:{element_bits_amount}={max_element}')

        self.bitstring.append(f'uint:5={counts_bits_amount}')

        for element in range(max_element + 1):
            try:
                element_count = frequency_table[element]
            except KeyError:
                element_count = 0
            self.bitstring.append(f'uint:{counts_bits_amount}={element_count}')

        ##### Write bitstring header
        words_amount = int(np.ceil(len(bitstring_AE)/16))
        missing_bits_amount = len(bitstring_AE)%16
        missing_bits_amount = 16 - missing_bits_amount if missing_bits_amount != 0 else 0

        ##### Write header
        if words_amount < 2**15:
            # Flag for using only 15 bits to write words amount.
            self.bitstring.append(f'0b0')
            self.bitstring.append(f'uint:15={words_amount}')
        else:
            # Flag for using 20 bits to write words amount.
            self.bitstring.append(f'0b1')
            self.bitstring.append(f'uint:20={words_amount}')
        
        ##### Write bitstring
        bitstring_AE += missing_bits_amount * '0'
        self.bitstring.append(f'0b{bitstring_AE}')

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