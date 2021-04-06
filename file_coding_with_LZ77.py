import os
import sys
import time
import argparse 
import numpy as np
import pandas as pd
from scipy.stats import entropy

from encoder import Encoder, menage_binary_file_path
from decoder import Decoder, menage_decoded_file_path


def print_process_duration(starting_time, ending_time, process_name):

    def check_plural(value, string):
        string += 's' if value > 1 else ''
        return string
    
    time_difference = int(ending_time - starting_time)
    seconds = time_difference % 60
    minutes = (time_difference // 60) % 60
    hours = (time_difference // 60) // 60
    
    hours_string = check_plural(hours, f'{hours} hour') + ', ' if hours else ''
    minutes_string = check_plural(minutes, f'{minutes} minute') + ', ' if minutes else ''
    space_string = 'and ' if (minutes or hours) else ''
    seconds_string = check_plural(seconds, f'{seconds} second') if seconds else '0 second'

    print(process_name + ' took ' + hours_string + minutes_string + seconds_string + '.')


if __name__ == "__main__":
    ##### Receives file to be compressed from command line.
    parser = argparse.ArgumentParser(description="Receives file to be encoded and some encoding options.")
    
    parser.add_argument('--file_to_compress', required=True, help='Path to file to be compressed.')
    parser.add_argument('--search_buffer_length', default=31, type=int, help='Buffer size with the already encoded symbols.')
    parser.add_argument('--look_ahead_buffer_length', default=15, type=int, help='Buffer size with symbols to be encoded.')
    parser.add_argument('-2', '--second_encoding_step', action='store_true', help='Flag to set a second encoding step.')
    parser.add_argument('-c', '--compare_diff_buffers', action='store_true', help='Evaluate performance with different buffer sizes.')
    parser.add_argument('--binary_file_path', required=False, help="Path to save binary file. "
                                                                   "If folders do not exist, they'll be created.")
    parser.add_argument('--decoded_file_path', required=False, help="Path to save decoded file. "
                                                                    "If folders do not exist, they'll be created.")

    ##### Read command line
    args = parser.parse_args(sys.argv[1:])

    ##### Menage binary file path
    menage_binary_file_path(args)

    ##### Menage decoded file path
    menage_decoded_file_path(args)

    ##### Define buffers sizes
    if args.compare_diff_buffers:
        buffer_sizes = [[15, 7], [31, 15], [63, 31]]
    else:
        buffer_sizes = [[args.search_buffer_length, args.look_ahead_buffer_length]]

    for buffers in buffer_sizes:
        ##### Printe buffer sizes:
        print(f"##### Search buffer size:     {buffers[0]};")
        print(f"##### Look ahead buffer size: {buffers[1]}.\n")
        
        ##### Encode source
        encoder = Encoder(args.file_to_compress)
        encoding_start = time.time()
        encoder.encode_sequence(*buffers, args.second_encoding_step)
        encoding_finish = time.time()
        binary_path, extension = os.path.splitext(args.binary_file_path)
        args.binary_file_path = binary_path + f'_{buffers[0]}_{buffers[1]}' + extension
        encoder.save_binary_file(args.binary_file_path)
        print_process_duration(encoding_start, encoding_finish, "Encoding Process")

        ##### Compute source entropy
        source_series = pd.Series(np.frombuffer(encoder.sequence, dtype=np.uint8))
        counts = source_series.value_counts()
        source_entropy = entropy(counts, base=2)

        ##### Compare entropy with achieved rate
        rate = encoder.compute_rate()

        print(f"The first-order source entropy is: {source_entropy:.5f} bits per symbol;")
        print(f"The achieved rate was            : {rate:.5f} bits per symbol.")

        ##### Decode source.
        decoder = Decoder(args.binary_file_path)
        decoding_start = time.time()
        decoder.decode_bitstring()
        decoding_finish = time.time()
        decoder.save_decoded_file(args.decoded_file_path)
        print_process_duration(decoding_start, decoding_finish, "Decoding Process")

        ##### Compute total time
        print_process_duration(encoding_start, decoding_finish, "Total Process")