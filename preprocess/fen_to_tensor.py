import numpy as np
import chess

PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        base = 0 if piece.color == chess.WHITE else 6
        ch = base + PIECE_TO_CHANNEL[piece.piece_type]
        tensor[ch, 7 - rank, file] = 1.0
    return tensor
