import numpy as np
from bitstring import BitArray


class LZ77():

    def __init__(self, search_buffer_size, look_ahead_buffer_size):
        self.search_buffer_size = search_buffer_size
        self.search_buffer = np.empty(search_buffer_size)
        self.look_ahead_buffer_size = look_ahead_buffer_size
        
    
    def encode_sequence(self, bytes_sequence):
        ##### Instantiate sequence bitstring
        self.bitstring = BitArray()

        ##### Get source's alphabet size.
        self.alphabet_length = len(np.unique(bytes_sequence))

        ##### Save sequence
        self.sequence = bytes_sequence

        ##### Create look_ahead_buffer
        self.look_ahead_buffer = self.sequence[:self.look_ahead_buffer_size]

        while len(self.sequence) is not 0:
            triple = self.__generate_triple()
            self.__update_buffers(triple[1])
            self.__save_triple_in_bitstring(triple)

        ##### Return encoded bitstring
        return self.bitstring


    ########## Private Methods

    def __generate_triple(self):
        
        ##### Search for a larger version of the sequence that starts the look
        #     ahead buffer while matches in the search buffer are found.
        for sequence_length in range(1, self.look_ahead_buffer_size + 1):
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
                while ((np.sum(self.search_buffer[seq_index: seq_index + sequence_length] == sequence_to_be_found) == sequence_length)
                                                                                 and (sequence_length <= self.look_ahead_buffer_size)):
                    sequence_length += 1
                    sequence_to_be_found = self.look_ahead_buffer[:sequence_length]
                match_length = sequence_length - 1
                symbol = sequence_to_be_found[-1]
                break
            else:
                last_founded_indexes = founded_indexes

        ##### Return triple.
        # self.sequence = self.sequence[match_length + 1:]
        return [offset, match_length, symbol]


    def __update_buffers(self, match_length):
        search_buffer_tail, self.sequence = np.split(self.sequence, [match_length + 1])
        self.search_buffer = np.concatenate((self.search_buffer[match_length + 1:], search_buffer_tail.astype(np.float64)))
        self.look_ahead_buffer = self.sequence[:self.look_ahead_buffer_size]


    def __save_triple_in_bitstring(self, triple):
        ##### Extract info from triple.
        offset, match_length, symbol = triple

        ##### Write info on bitstream.
        self.bitstring.append(f'uint:8={offset}')
        self.bitstring.append(f'uint:8={match_length}')
        self.bitstring.append(f'uint:8={symbol}')

        ##### Actually, this approach is applied for the decoder.
        # joint_buffer = self.search_buffer + self.look_ahead_buffer
        # first_idx = self.search_buffer_size - offset
        # last_idx = self.first_idx + match_length



    ########## Auxiliary Methods

    def __rolling_window(self, window_size):
        shape = self.search_buffer.shape[:-1] + (self.search_buffer.shape[-1] - window_size + 1, window_size)
        strides = self.search_buffer.strides + (self.search_buffer.strides[-1],)
        return np.lib.stride_tricks.as_strided(self.search_buffer, shape=shape, strides=strides)



text_to_encode = 'aboboraeboba'
sequence = np.frombuffer(text_to_encode.encode(), dtype=np.int8)
encoder = LZ77(7,5)
sequence_bs = encoder.encode_sequence(sequence)