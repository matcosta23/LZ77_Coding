import numpy as np
from bitstring import BitArray


class LZ77():

    def __init__(self, search_buffer_size, look_ahead_buffer_size):
        self.search_buffer_size = search_buffer_size
        self.search_buffer = np.empty(search_buffer_size)
        self.look_ahead_buffer_size = look_ahead_buffer_size
        

    def read_sequence(self, bytes_sequence):
        ##### Save sequence
        self.sequence = bytes_sequence

    
    def encode_sequence(self):
        ##### Generate triples
        self.generate_triples()

        ##### Write triples to the bitstream
        self.write_triples_in_bitstring()

        ##### Return bitstring
        return self.get_bitstring()


    def generate_triples(self, ):
        ##### Create look_ahead_buffer
        self.look_ahead_buffer = self.sequence[:self.look_ahead_buffer_size]

        ##### Create list for saving triples.
        self.triples = []

        while len(self.sequence) is not 0:
            triple = self.__generate_triple()
            self.__update_buffers(triple[1])
            self.triples.append(triple)

        return self.triples    


    def write_triples_in_bitstring(self):
        self.triples = np.array(self.triples)

        ##### Get maximum offset and match length values.
        max_offset = self.triples[:, 0].max()
        max_match_length = self.triples[:, 1].max()

        ##### Get amount of bits required to send offset and match length.
        offset_bits_amount = len(bin(max_offset)[2:])
        match_length_bits_amount = len(bin(max_match_length)[2:])

        ##### Instantiate bitstring with header.
        # NOTE: We use 5 bits to write the amount of bits that encode offset and match length info.
        self.bitstring = BitArray(f'uint:5={offset_bits_amount}, uint:5={match_length_bits_amount}')

        ##### Write triples in bitstring
        for triple in self.triples:
            self.bitstring.append(f'uint:{offset_bits_amount}={triple[0]}')
            self.bitstring.append(f'uint:{match_length_bits_amount}={triple[1]}')
            self.bitstring.append(f'uint:8={triple[2]}')


    def get_bitstring(self):
        return self.bitstring


    ########## Private Methods


    def __generate_triple(self):
        ##### Define empty variables
        offset = match_length = symbol = None
        
        ##### Search for a larger version of the sequence that starts the look
        #     ahead buffer while matches in the search buffer are found.
        for sequence_length in range(1, self.look_ahead_buffer_size):
            sequence_to_be_found = self.look_ahead_buffer[:sequence_length]
            ##### Get indexes where the current sequence is founded in the search buffer.
            founded_indexes = np.where(np.all(self.__rolling_window(sequence_length) == sequence_to_be_found, axis=1) == True)[0]

            ##### If there are no indexes, it means that the sequence was found in the search buffer.
            if founded_indexes.shape[0] is 0:
                ##### If no match is found in the first element, it means that this element is not available in the search buffer.
                if sequence_length is 1:
                    offset = match_length = 0
                    symbol = self.look_ahead_buffer[0]
                    break
                ##### However, if this happened for longer sequences, it means that the
                #     sequence with one element less was found in the search buffer.
                else:
                    seq_index = last_founded_indexes[0]
                    offset = self.search_buffer_size - seq_index
                    match_length = sequence_length - 1
                    symbol = sequence_to_be_found[-1]
                    break

            ##### If the sequence was found at only one index, it is sufficient
            #     to check the size of the match from that index.
            elif founded_indexes.shape[0] is 1:
                seq_index = founded_indexes[0]
                offset = self.search_buffer_size - seq_index
                ##### Grow the sequence size as long as the match is true.
                sequence_length += 1
                sequence_to_be_found = self.look_ahead_buffer[:sequence_length]
                while ((np.sum(self.search_buffer[seq_index: seq_index + sequence_length] == sequence_to_be_found) == sequence_length)
                            and (sequence_length < len(self.look_ahead_buffer))):
                    sequence_length += 1
                    sequence_to_be_found = self.look_ahead_buffer[:sequence_length]
                match_length = sequence_length - 1
                symbol = self.look_ahead_buffer[match_length]
                break
        
            else:
                last_founded_indexes = founded_indexes

        ##### If sequence length has achieved the greatest value, you assign the largest
        #     possible offset and encode the last symbol of the look ahead buffer.
        if (offset and match_length and symbol) is None:
            offset = self.search_buffer_size - founded_indexes[0]
            match_length = sequence_length
            symbol = self.look_ahead_buffer[-1]

        ##### Return triple.
        return [offset, match_length, symbol]


    def __update_buffers(self, match_length):
        search_buffer_tail, self.sequence = np.split(self.sequence, [match_length + 1])
        self.search_buffer = np.concatenate((self.search_buffer[match_length + 1:], search_buffer_tail.astype(np.float64)))
        self.look_ahead_buffer = self.sequence[:self.look_ahead_buffer_size]


    ########## Auxiliary Methods

    def __rolling_window(self, window_size):
        shape = self.search_buffer.shape[:-1] + (self.search_buffer.shape[-1] - window_size + 1, window_size)
        strides = self.search_buffer.strides + (self.search_buffer.strides[-1],)
        return np.lib.stride_tricks.as_strided(self.search_buffer, shape=shape, strides=strides)