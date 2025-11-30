// engine/src/NNBoardTensor.cpp

#include "engine/NNBoardTensor.h"
#include "engine/board.h" // 실제 경로에 맞게 수정 (예: "board.h")
#include "engine/types.h" // isWhite, pieceType, PieceType, Color 정의

#include <cstring> // std::memset

namespace chess {

// 채널 인덱스 (fen_to_tensor.py와 1:1 대응)
static constexpr int WHITE_PAWN = 0;
static constexpr int WHITE_KNIGHT = 1;
static constexpr int WHITE_BISHOP = 2;
static constexpr int WHITE_ROOK = 3;
static constexpr int WHITE_QUEEN = 4;
static constexpr int WHITE_KING = 5;

static constexpr int BLACK_PAWN = 6;
static constexpr int BLACK_KNIGHT = 7;
static constexpr int BLACK_BISHOP = 8;
static constexpr int BLACK_ROOK = 9;
static constexpr int BLACK_QUEEN = 10;
static constexpr int BLACK_KING = 11;

static constexpr int SIDE_TO_MOVE = 12;
static constexpr int CASTLE_WK = 13;
static constexpr int CASTLE_WQ = 14;
static constexpr int CASTLE_BK = 15;
static constexpr int CASTLE_BQ = 16;
static constexpr int EP_FILE = 17;

static constexpr int NUM_CHANNELS = 18;
static constexpr int BOARD_SIZE = 8;

// 편의 함수: (ch, rank, file) → 1D 인덱스
static inline int idx3(int ch, int rank, int file) {
  // ch: 0..17, rank: 0..7 (1~8 rank), file: 0..7 (a~h)
  return ch * BOARD_SIZE * BOARD_SIZE + rank * BOARD_SIZE + file;
}

void boardToTensor(const Board &board,
                   uint8_t outCHW[NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE]) {
  // 전체 0으로 초기화
  std::memset(outCHW, 0,
              NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE * sizeof(uint8_t));

  // 1) 기물 채널 (12채널)
  //   - Board::squares: 0=a1, 1=b1, ... 7=h1, 8=a2 ... 63=h8
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.pieceAt(sq);
    if (p == NO_PIECE)
      continue;

    int file = sq % 8; // 0=a ... 7=h
    int rank = sq / 8; // 0=1 ... 7=8

    int ch = -1;

    bool white = isWhite(p);
    PieceType pt = pieceType(p);

    if (white) {
      switch (pt) {
      case PAWN:
        ch = WHITE_PAWN;
        break;
      case KNIGHT:
        ch = WHITE_KNIGHT;
        break;
      case BISHOP:
        ch = WHITE_BISHOP;
        break;
      case ROOK:
        ch = WHITE_ROOK;
        break;
      case QUEEN:
        ch = WHITE_QUEEN;
        break;
      case KING:
        ch = WHITE_KING;
        break;
      default:
        break;
      }
    } else {
      switch (pt) {
      case PAWN:
        ch = BLACK_PAWN;
        break;
      case KNIGHT:
        ch = BLACK_KNIGHT;
        break;
      case BISHOP:
        ch = BLACK_BISHOP;
        break;
      case ROOK:
        ch = BLACK_ROOK;
        break;
      case QUEEN:
        ch = BLACK_QUEEN;
        break;
      case KING:
        ch = BLACK_KING;
        break;
      default:
        break;
      }
    }

    if (ch >= 0) {
      int idx = idx3(ch, rank, file);
      outCHW[idx] = 1;
    }
  }

  // 2) side to move 채널
  // fen_to_tensor.py:
  //   if board.turn == WHITE: arr[SIDE_TO_MOVE, :, :] = 1
  if (board.stm == WHITE) {
    for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int f = 0; f < BOARD_SIZE; ++f) {
        outCHW[idx3(SIDE_TO_MOVE, r, f)] = 1;
      }
    }
  }

  // 3) castling rights (WK, WQ, BK, BQ)
  // fen_to_tensor.py와 동일: 해당 권리가 있으면 전체 8x8에 1
  if (board.castleWK) {
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int f = 0; f < BOARD_SIZE; ++f)
        outCHW[idx3(CASTLE_WK, r, f)] = 1;
  }

  if (board.castleWQ) {
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int f = 0; f < BOARD_SIZE; ++f)
        outCHW[idx3(CASTLE_WQ, r, f)] = 1;
  }

  if (board.castleBK) {
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int f = 0; f < BOARD_SIZE; ++f)
        outCHW[idx3(CASTLE_BK, r, f)] = 1;
  }

  if (board.castleBQ) {
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int f = 0; f < BOARD_SIZE; ++f)
        outCHW[idx3(CASTLE_BQ, r, f)] = 1;
  }

  // 4) en-passant file 채널
  // fen_to_tensor.py:
  //   if board.ep_square is not None:
  //       ep_file = chess.square_file(board.ep_square)
  //       arr[EP_FILE, :, ep_file] = 1
  if (board.epSquare >= 0 && board.epSquare < 64) {
    int epFile = board.epSquare % 8; // 0=a ... 7=h
    for (int r = 0; r < BOARD_SIZE; ++r) {
      outCHW[idx3(EP_FILE, r, epFile)] = 1;
    }
  }
}

} // namespace chess
