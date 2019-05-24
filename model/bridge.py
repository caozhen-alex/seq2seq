import torch
import torch.nn as nn

class Bridge(nn.Module):

    def __init__(self, rnn_type, hidden_size, bidirectional):
        super(Bridge, self).__init__()
        self._bidirectional = bidirectional
        encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self._encoder_output_projection = nn.Linear(encoder_output_size, hidden_size)
        self._rnn_type = rnn_type
        if self._rnn_type == 'LSTM':
            self._hidden_projection = nn.Linear(encoder_output_size, hidden_size)
            self._cell_projection = nn.Linear(encoder_output_size, hidden_size)
        elif self._rnn_type == 'GRU':
            self._hidden_projection = nn.Linear(encoder_output_size, hidden_size)
        else:
            raise ValueError('No Supporting.')

    def forward(self, encoder_output, final_encoder_states):
        # encoder_output: Tensor (batch_size, time_step, hidden_size * num_directions)
        # final_encoder_hidden: Tensor (num_layers * num_directions, batch_size, hidden_size)
        # final_encoder_cell: Tensor (num_layers * num_directions, batch_size, hidden_size)

        # encoder_memory: Tensor(batch_size, time_step, hidden_size)
        # init_decoder_states: (init_decoder_hidden, init_decoder_cell): Tensor(0.5 *  num_layers * num_directions, batch_size, 2 * hidden_size)--->Tensor(0.5 *  num_layers * num_directions, batch_size, hidden_size)
        encoder_memory = self._encoder_output_projection(encoder_output)
        if self._rnn_type == 'LSTM':    # LSTM
            if self._bidirectional:
                final_encoder_states = (
                    torch.cat(
                        final_encoder_states[0].chunk(chunks=2, dim=0),  # Tensor(num_layers * num_directions, batch_size, hidden_size) ---> tuple of tensors: 2 * Tensor(0.5 *  num_layers * num_directions, batch_size, hidden_size)
                        dim=2
                    ),  # 2 * Tensor(0.5 *  num_layers * num_directions, batch_size, hidden_size) ---> Tensor(0.5 *  num_layers * num_directions, batch_size, 2 * hidden_size)
                    torch.cat(
                        final_encoder_states[1].chunk(chunks=2, dim=0),
                        dim=2
                    )
                )
            init_decoder_states = (
                torch.stack([
                    self._hidden_projection(hidden) for hidden in final_encoder_states[0]  # hidden is the subtensor of final_encoer_states[0] at dim = 0
                ], dim=0),  # torch.stack: Concatenates sequence of tensors along a new dimension.
                torch.stack([
                    self._cell_projection(cell) for cell in final_encoder_states[1]
                ], dim=0)
            )
        else:   # GRU
            if self._bidirectional:
                final_encoder_states = torch.cat(
                    final_encoder_states.chunk(chunks=2, dim=0),
                    dim=2
                )
            init_decoder_states = torch.stack([
                self._hidden_projection(hidden) for hidden in final_encoder_states
            ], dim=0)
        return encoder_memory, init_decoder_states
