import numpy as np
import chess

# 채널 인덱스 정의
WHITE_PAWN   = 0
WHITE_KNIGHT = 1
WHITE_BISHOP = 2
WHITE_ROOK   = 3
WHITE_QUEEN  = 4
WHITE_KING   = 5

BLACK_PAWN   = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_ROOK   = 9
BLACK_QUEEN  = 10
BLACK_KING   = 11

SIDE_TO_MOVE = 12
CASTLE_WK    = 13
CASTLE_WQ    = 14
CASTLE_BK    = 15
CASTLE_BQ    = 16
EP_FILE      = 17

NUM_CHANNELS = 18


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    chess.Board -> (18, 8, 8) uint8
    12채널: 기물 (흰/검 6종)
    1채널: side to move
    4채널: castling rights (WK,WQ,BK,BQ)
    1채널: en-passant file
    """
    arr = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.uint8)

    # 1) 기물 채널
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        file = chess.square_file(sq)  # 0=a ... 7=h
        rank = chess.square_rank(sq)  # 0=1 ... 7=8

        ch = -1
        if piece.color == chess.WHITE:
            if piece.piece_type == chess.PAWN:
                ch = WHITE_PAWN
            elif piece.piece_type == chess.KNIGHT:
                ch = WHITE_KNIGHT
            elif piece.piece_type == chess.BISHOP:
                ch = WHITE_BISHOP
            elif piece.piece_type == chess.ROOK:
                ch = WHITE_ROOK
            elif piece.piece_type == chess.QUEEN:
                ch = WHITE_QUEEN
            elif piece.piece_type == chess.KING:
                ch = WHITE_KING
        else:
            if piece.piece_type == chess.PAWN:
                ch = BLACK_PAWN
            elif piece.piece_type == chess.KNIGHT:
                ch = BLACK_KNIGHT
            elif piece.piece_type == chess.BISHOP:
                ch = BLACK_BISHOP
            elif piece.piece_type == chess.ROOK:
                ch = BLACK_ROOK
            elif piece.piece_type == chess.QUEEN:
                ch = BLACK_QUEEN
            elif piece.piece_type == chess.KING:
                ch = BLACK_KING

        if ch >= 0:
            arr[ch, rank, file] = 1

    # 2) side to move
    if board.turn == chess.WHITE:
        arr[SIDE_TO_MOVE, :, :] = 1

    # 3) castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        arr[CASTLE_WK, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        arr[CASTLE_WQ, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        arr[CASTLE_BK, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        arr[CASTLE_BQ, :, :] = 1

    # 4) en-passant file
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        arr[EP_FILE, :, ep_file] = 1

    return arr


if __name__ == "__main__":
    # 간단 테스트
    b = chess.Board()  # startpos
    t = board_to_tensor(b)
    print("shape:", t.shape)            # (18, 8, 8)
    print("dtype:", t.dtype)            # uint8
    print("sum per channel:", t.reshape(18, -1).sum(axis=1))
